"""
Triton FP8 Fused MoE kernel — v22 (v20 base, v21 removed)

Pipeline (all GPU, no CPU sync):
  0. _init_workspace         — zero routing buffers + sorted_weights; fill token_slot_map
                                with gcap sentinel; 1 kernel, grid sized by T*TOP_K
  1. _routing_kernel         — [T,8] expert indices + weights + fused expert counting
  2. _prefix_sum             — expert_offsets [E_LOCAL+1] + write_ptrs (1 CTA, registers)
  3. _scatter_sorted_tokens  — sorted_token_ids, sorted_weights, token_slot_map
  4. _grouped_gemm1_swiglu   — FP8×FP8→fp32 with inline token gather (note: SwiGLU
                                is NOT fused; the kernel name is historical — see step 5)
  5. _swiglu_to_fp16_scaled  — silu(up)*gate on fp32 GEMM1 output; per-row × per-128-K
                                FP16 quantization for GEMM2
  6. _grouped_gemm2          — FP16×FP8→fp32, plain tl.store to gemm2_out [gcap, H]
  7. _weighted_output        — token-parallel weighted gather: gemm2_out[slot,:] *
                                sorted_weights[slot] → output [T, H] bf16

v22 wraps the entire entry point in CUDA Graph capture/replay (see `kernel()` at the
bottom).  flashinfer-bench reuses the same input pointers across all 100 iterations of
a trial, so we capture once (after an eager warmup that fills the @triton.autotune
caches) and replay for the remaining 99 calls — collapsing ~95µs of inter-kernel
host-launch idle into a single cudaGraphLaunch.

v21's fused atomic-add scatter (_grouped_gemm2_weighted) has been removed.  Profiling
showed that atomic_add into out_f32 [T, H] from GEMM2 tiles regressed the two largest
workloads by ~9% (5e8dc11c: 4.55 ms → 4.98 ms) despite improving small workloads.
v20's separate _weighted_output gather (plain HBM reads, no atomics) is restored.

Per-version history and benchmark numbers live in CHANGELOG.md.
"""

import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Model constants
# ─────────────────────────────────────────────────────────────────────────────
H           = 7168
I           = 2048
E_GLOBAL    = 256
E_LOCAL     = 32
TOP_K       = 8
N_GROUP     = 8
TOPK_GROUP  = 4
QUANT_BLOCK = 128

BLOCK_N = 128
BLOCK_K = 128

# ─────────────────────────────────────────────────────────────────────────────
# Persistent workspace cache (eliminates per-call allocations and CPU syncs)
# ─────────────────────────────────────────────────────────────────────────────
_ws_cache: dict = {}

# CUDA Graph cache — keyed by (T, *all input/output data_ptrs).  flashinfer-bench
# reuses the same input tensors across the 100 timed iterations of a trial, so
# replay is hit ~99% of the time; a fresh trial allocates new tensors and falls
# through to the (re)capture path.  Captured graph eliminates ~95µs of
# inter-kernel host launch latency between the 8 short kernels in this pipeline.
_graph_cache: dict = {}


def _ensure_workspace(T: int, device):
    """Return (creating if needed) workspace tensors for batch size T."""
    key = (T, str(device))
    if key in _ws_cache:
        return _ws_cache[key]
    gcap = max(T * 2, 128)   # capacity: 2× expected total_tokens
    ws = {
        'topk_idx':         torch.empty((T, TOP_K),    dtype=torch.int32,   device=device),
        'topk_weights':     torch.empty((T, TOP_K),    dtype=torch.float32, device=device),
        'expert_counts':    torch.zeros(E_LOCAL,       dtype=torch.int32,   device=device),
        'expert_offsets':   torch.zeros(E_LOCAL + 1,   dtype=torch.int32,   device=device),
        'write_ptrs':       torch.empty(E_LOCAL,       dtype=torch.int32,   device=device),
        'sorted_token_ids': torch.zeros(gcap,          dtype=torch.int32,   device=device),
        'sorted_weights':   torch.zeros(gcap,          dtype=torch.float32, device=device),
        # token_slot_map [T, TOP_K]: maps (token, k) → workspace slot in sorted layout.
        # Filled with gcap sentinel each call by _init_workspace; local expert slots
        # are overwritten by _scatter_sorted_tokens.  _weighted_output uses slot < gcap
        # to skip non-local (sentinel) entries.
        'token_slot_map':   torch.empty((T, TOP_K),    dtype=torch.int32,   device=device),
        'gemm1_out':        torch.zeros((gcap, 2 * I), dtype=torch.float32, device=device),
        'swiglu_fp16':      torch.empty((gcap, I),     dtype=torch.float16, device=device),
        'swiglu_scale_a':   torch.empty((I // QUANT_BLOCK, gcap), dtype=torch.float32, device=device),
        # gemm2_out [gcap, H] fp32: plain tl.store target for GEMM2.
        # Each valid workspace slot gets exactly one store; _weighted_output then
        # gathers and weights these into the [T, H] bf16 output.
        'gemm2_out':        torch.empty((gcap, H),     dtype=torch.float32, device=device),
        'gcap':             gcap,
    }
    _ws_cache[key] = ws
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# Autotune configs
# ─────────────────────────────────────────────────────────────────────────────
_GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': 128, 'BLOCK_K': 128},
                  num_warps=NW, num_stages=NS)
    for BM in [16, 32, 64, 128]
    for NW in [4, 8]
    for NS in [3, 4, 5]
]

_GEMM1_CONFIGS = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': 128, 'BLOCK_K': 128},
                  num_warps=NW, num_stages=NS)
    for BM in [16, 32, 64, 128]
    for NW in [4, 8]
    for NS in [3, 4, 5]
]


@triton.autotune(configs=_GEMM1_CONFIGS, key=['N', 'K', 'total_tokens'])
@triton.jit
def _grouped_gemm1_swiglu(
    # token_ids: sorted_token_ids [total_tokens] int32 — maps workspace slot → original token row
    token_ids_ptr,
    # A: raw hidden states [T, K] FP8  (NOT pre-gathered; indexed via token_ids)
    A_ptr, SA_ptr,
    # B: W1 weights [E_LOCAL, 2*I, K] FP8  (N = 2*I)
    B_ptr, SB_ptr,
    # C: GEMM1 output [total_tokens, 2*I] f32
    C_ptr,
    # expert token ranges
    expert_offsets_ptr,       # [E_LOCAL+1] int32
    # dims
    N, K,                     # N = 2*I, K = H
    total_tokens,             # autotune key — total rows across all experts
    # strides for A [T, K]
    stride_am, stride_ak,
    # strides for SA [K//128, T]
    stride_sA_kb, stride_sA_m,
    # strides for B [E_LOCAL, 2*I, K]
    stride_be, stride_bn, stride_bk,
    # strides for SB [E_LOCAL, 2*I//128, K//128]
    stride_sB_e, stride_sB_nb, stride_sB_kb,
    # strides for C [total_tokens, 2*I]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m_global = tl.program_id(0)
    pid_n        = tl.program_id(1)

    pid_e   = 0
    pid_m   = 0
    cum_m   = 0
    for e in tl.static_range(32):
        e_s       = tl.load(expert_offsets_ptr + e).to(tl.int32)
        e_e       = tl.load(expert_offsets_ptr + e + 1).to(tl.int32)
        m_tiles_e = (e_e - e_s + BLOCK_M - 1) // BLOCK_M
        new_cum   = cum_m + m_tiles_e
        found     = (pid_m_global >= cum_m) & (pid_m_global < new_cum)
        pid_e     = tl.where(found, e, pid_e)
        pid_m     = tl.where(found, pid_m_global - cum_m, pid_m)
        cum_m     = new_cum

    if pid_m_global >= cum_m:
        return   # over-launched CTA (fixed grid); no actual data here

    e_start = tl.load(expert_offsets_ptr + pid_e).to(tl.int32)
    e_end   = tl.load(expert_offsets_ptr + pid_e + 1).to(tl.int32)
    M_e     = e_end - e_start

    rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rm       = e_start + rm_local
    rn       = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ── Inline gather: resolve workspace slots → original token row indices ──
    # Loaded once here, outside the K-loop, replacing the pre-kernel aten::index
    # that materialized the full [gcap, H] sorted_A tensor.
    # Out-of-bounds slots (rm_local >= M_e) load token 0 — harmless since those
    # rows are masked out at the store below.
    tok = tl.load(token_ids_ptr + rm, mask=rm_local < M_e, other=0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(K // BLOCK_K):
        rk = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        # Index into raw hidden_states[tok, :] instead of pre-gathered sorted_A[rm, :]
        a = tl.load(
            A_ptr + tok[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm_local[:, None] < M_e) & (rk[None, :] < K), other=0.0,
        )
        # Index into raw hidden_states_scale[kb, tok] instead of sorted_A_scale[kb, rm]
        sa = tl.load(
            SA_ptr + kb * stride_sA_kb + tok * stride_sA_m,
            mask=rm_local < M_e, other=1.0,
        )
        b = tl.load(
            B_ptr + pid_e * stride_be + rn[:, None] * stride_bn + rk[None, :] * stride_bk,
            mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0,
        )
        sb = tl.load(SB_ptr + pid_e * stride_sB_e + pid_n * stride_sB_nb + kb * stride_sB_kb)

        # FP8 × FP8 → float32 using FP8 tensor cores; apply block scales after.
        partial = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
        acc = acc + partial * sa[:, None] * sb

    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm_local[:, None] < M_e) & (rn[None, :] < N),
    )


@triton.jit
def _swiglu_to_fp16_scaled(
    X_ptr,           # [total_tokens, 2*I] FP32  (gemm1_out, read-only)
    Y_ptr,           # [total_tokens, I]   FP16  (output, per-row × per-128 scaled)
    SA_ptr,          # [I//BLOCK_I, total_tokens] FP32  (per-row × per-128 scale)
    total_tokens,
    stride_sA_kb,    # = total_tokens (rows are K-blocks)
    stride_sA_m,     # = 1            (cols are token-rows)
    I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """SwiGLU + per-row, per-128-K-block FP16 quantization.

    For each (BLOCK_M tokens × BLOCK_I=128 contiguous K-elems) tile:
      z       = sigmoid(up) * up * gate       (FP32)
      row_max = max(|z|, axis=1)              (FP32, [BLOCK_M])
      scale   = max(row_max, eps) / 32000     (target |z_fp16| ≤ 32000)
      store   z / scale[:, None]   as FP16
      store   scale                 to SA[pid_i, rm]

    Aligning the scale to BLOCK_K=128 along the K dimension keeps the scale
    constant across each reduction step inside `_grouped_gemm2`, so it can be
    factored outside `tl.dot` and applied to the FP32 accumulator without
    losing FP16 tensor-core throughput.
    """
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ri = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = rm < total_tokens
    mask_i = ri < I
    mask   = mask_m[:, None] & mask_i[None, :]

    gate = tl.load(X_ptr + rm[:, None] * (2 * I) + ri[None, :],
                   mask=mask, other=0.0)
    up   = tl.load(X_ptr + rm[:, None] * (2 * I) + (ri[None, :] + I),
                   mask=mask, other=0.0)

    z = tl.sigmoid(up) * up * gate                  # [BLOCK_M, BLOCK_I] FP32

    # Per-row, per-128-K-block dynamic scale. 32000 leaves 2× headroom from
    # FP16 max=65504, so any rounding/cast bumps still stay finite.
    row_max = tl.max(tl.abs(z), axis=1)             # [BLOCK_M]
    scale   = tl.maximum(row_max, 1e-30) / 32000.0  # [BLOCK_M]

    z_fp16 = (z / scale[:, None]).to(tl.float16)    # values in [-32000, 32000]

    tl.store(Y_ptr + rm[:, None] * I + ri[None, :], z_fp16, mask=mask)
    tl.store(SA_ptr + pid_i * stride_sA_kb + rm * stride_sA_m,
             scale, mask=mask_m)


@triton.autotune(configs=_GEMM_CONFIGS, key=['N', 'K', 'total_tokens'])
@triton.jit
def _grouped_gemm2(
    A_ptr, SA_ptr,            # A: [total_tokens, K] FP16 ; SA: [K//128, total_tokens] FP32
    B_ptr, SB_ptr,            # B: FP8 weights [E_LOCAL, H, I], SB: per-128-block scales
    C_ptr,                    # C: [gcap, H] f32 — plain store (no atomic), one tile per slot
    expert_offsets_ptr,
    N, K,
    total_tokens,
    stride_am, stride_ak,
    stride_sA_kb, stride_sA_m,
    stride_be, stride_bn, stride_bk,
    stride_sB_e, stride_sB_nb, stride_sB_kb,
    stride_cm, stride_cn,     # strides for C [gcap, H]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grouped GEMM2: FP16×FP8→FP32, plain tl.store to gemm2_out.

    Each workspace slot rm is owned by exactly one CTA tile, so tl.store is
    safe and avoids the atomic_add contention that v21 introduced.
    _weighted_output then gathers from gemm2_out via token_slot_map.
    """
    pid_m_global = tl.program_id(0)
    pid_n        = tl.program_id(1)

    pid_e   = 0
    pid_m   = 0
    cum_m   = 0
    for e in tl.static_range(32):
        e_s       = tl.load(expert_offsets_ptr + e).to(tl.int32)
        e_e       = tl.load(expert_offsets_ptr + e + 1).to(tl.int32)
        m_tiles_e = (e_e - e_s + BLOCK_M - 1) // BLOCK_M
        new_cum   = cum_m + m_tiles_e
        found     = (pid_m_global >= cum_m) & (pid_m_global < new_cum)
        pid_e     = tl.where(found, e, pid_e)
        pid_m     = tl.where(found, pid_m_global - cum_m, pid_m)
        cum_m     = new_cum

    if pid_m_global >= cum_m:
        return   # over-launched CTA (fixed grid)

    e_start = tl.load(expert_offsets_ptr + pid_e).to(tl.int32)
    e_end   = tl.load(expert_offsets_ptr + pid_e + 1).to(tl.int32)
    M_e     = e_end - e_start

    rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rm       = e_start + rm_local
    rn       = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_m  = rm_local < M_e

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(K // BLOCK_K):
        rk = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        # A is FP16 (per-row × per-128-K-block scaled SwiGLU output).
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=valid_m[:, None] & (rk[None, :] < K), other=0.0,
        )
        sa = tl.load(
            SA_ptr + kb * stride_sA_kb + rm * stride_sA_m,
            mask=valid_m, other=0.0,
        )

        # B is FP8 → FP16 directly (lossless: FP8 max=448 fits, 3-bit
        # mantissa fits in 10-bit FP16). Per-128 SB factored outside dot.
        b = tl.load(
            B_ptr + pid_e * stride_be + rn[:, None] * stride_bn + rk[None, :] * stride_bk,
            mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0,
        )
        sb     = tl.load(SB_ptr + pid_e * stride_sB_e + pid_n * stride_sB_nb + kb * stride_sB_kb)
        b_fp16 = b.to(tl.float16)

        # FP16 × FP16 → FP32 tensor core (HMMA on Hopper).
        # Both block scales are constant across the reduction → applied to acc.
        partial = tl.dot(a, tl.trans(b_fp16), out_dtype=tl.float32)
        acc = acc + partial * sa[:, None] * sb

    # Plain store: each workspace slot is written by exactly one tile.
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=valid_m[:, None] & (rn[None, :] < N),
    )


@triton.jit
def _weighted_output(
    gemm2_out_ptr,       # [gcap, H]    f32  — GEMM2 output indexed by workspace slot
    sorted_weights_ptr,  # [gcap]       f32  — routing weight per workspace slot
    token_slot_map_ptr,  # [T, TOP_K]   int32 — (token, k) → workspace slot; gcap = sentinel
    output_ptr,          # [T, H]       bf16 — final weighted output
    T, H, gcap,
    BLOCK_N: tl.constexpr,
):
    """Token-parallel weighted gather: produce one output token row per CTA-row.

    Grid = (T, H // BLOCK_N).  Each CTA owns one (output_token, BLOCK_N-col) tile.
    Iterates over the TOP_K=8 expert slots for this token; for each valid slot
    (slot < gcap), loads gemm2_out[slot, rn] and sorted_weights[slot], accumulates
    in fp32, and stores bfloat16 directly to output.

    No atomics needed: each output token is owned by exactly one CTA (per column
    block), and the inner loop is sequential over the token's 8 expert slots.
    Replaces _cast_f32_to_bf16 (v21) and the former index_add_ path (pre-v16).
    """
    pid_t = tl.program_id(0)   # token index
    pid_n = tl.program_id(1)   # column block

    if pid_t >= T:
        return

    rn     = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < H

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in tl.static_range(8):   # TOP_K = 8
        slot      = tl.load(token_slot_map_ptr + pid_t * 8 + k)
        is_valid  = slot < gcap
        slot_safe = tl.where(is_valid, slot, 0)

        w   = tl.load(sorted_weights_ptr + slot_safe,
                      mask=is_valid, other=0.0)
        val = tl.load(gemm2_out_ptr + slot_safe * H + rn,
                      mask=is_valid & mask_n, other=0.0)
        acc = acc + val * w

    tl.store(output_ptr + pid_t * H + rn, acc.to(tl.bfloat16), mask=mask_n)


# ─────────────────────────────────────────────────────────────────────────────
# v11 routing kernels
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _routing_kernel(
    logits_ptr,           # [T, 256] f32
    bias_ptr,             # [256] f32
    topk_idx_ptr,         # [T, 8] int32   (output)
    topk_weights_ptr,     # [T, 8] f32     (output)
    expert_counts_ptr,    # [E_LOCAL] int32 (output, zeroed before call; fused counting)
    T, local_start, routed_scaling_factor,
):
    """One CTA per token. Fuses sigmoid+bias, group scoring, top-K selection,
    and expert token counting into a single kernel.

    Merges the former `_count_expert_tokens` pass: after selecting the top-8
    experts, `topk_idx_arr` is still live in registers, so we atomic-add each
    local expert's slot directly — no global-memory round-trip needed.

    The global barrier between counting and prefix-sum still exists as an
    inter-kernel sync (prefix_sum waits for all T CTAs of this kernel), and the
    barrier between prefix-sum and scatter is likewise unavoidable: these
    require grid-wide synchronisation (CUDA cooperative kernels) that Triton
    does not expose. Only the routing→counting barrier is eliminated here.
    """
    pid = tl.program_id(0)
    if pid >= T:
        return

    # E_GLOBAL=256, N_GROUP=8, GROUP_SIZE=32, TOPK_GROUP=4, TOP_K=8
    e_ids = tl.arange(0, 256)   # [256]
    k_ids = tl.arange(0, 8)     # [8]

    logits = tl.load(logits_ptr + pid * 256 + e_ids).to(tl.float32)
    bias   = tl.load(bias_ptr + e_ids).to(tl.float32)
    s      = tl.sigmoid(logits)   # unbiased sigmoid — used for weight normalization
    s_bias = s + bias             # biased — used for routing decisions

    # ── Group scores: sum of top-2 sigmoid values per group ──────────────────
    group_scores = tl.zeros((8,), dtype=tl.float32) - 1e9

    for g in tl.static_range(8):        # N_GROUP = 8
        gs = g * 32                     # group start expert index (GROUP_SIZE = 32)
        in_grp = (e_ids >= gs) & (e_ids < gs + 32)
        g_vals = tl.where(in_grp, s_bias, -1e9)
        m1     = tl.max(g_vals)
        g_val2 = tl.where(g_vals >= m1, -1e9, g_vals)
        m2     = tl.max(g_val2)
        group_scores = tl.where(tl.arange(0, 8) == g, m1 + m2, group_scores)

    # ── Select top-4 groups ───────────────────────────────────────────────────
    gs_tmp         = group_scores
    selected_gmask = tl.zeros((8,), dtype=tl.float32)

    for _ in tl.static_range(4):        # TOPK_GROUP = 4
        best_gs   = tl.max(gs_tmp)
        is_best_g = gs_tmp >= best_gs
        selected_gmask = tl.where(is_best_g, 1.0, selected_gmask)
        gs_tmp         = tl.where(is_best_g, -1e9, gs_tmp)

    # ── Expand group mask to expert dimension ─────────────────────────────────
    expert_sel = tl.zeros((256,), dtype=tl.float32)

    for g in tl.static_range(8):
        gs    = g * 32
        g_sel = tl.sum(tl.where(tl.arange(0, 8) == g, selected_gmask, 0.0))
        in_g  = (e_ids >= gs) & (e_ids < gs + 32)
        expert_sel = tl.where(in_g, g_sel, expert_sel)

    # ── Find top-8 experts via 8 rounds of max + mask ─────────────────────────
    cur_vals     = tl.where(expert_sel > 0.5, s_bias, -1e9)
    topk_idx_arr = tl.zeros((8,), dtype=tl.int32)
    topk_w_arr   = tl.zeros((8,), dtype=tl.float32)

    for k in tl.static_range(8):        # TOP_K = 8
        best_val  = tl.max(cur_vals)
        is_best_e = cur_vals >= best_val
        # argmax as min-of-indices where score is best (lowest index on tie)
        best_idx  = tl.min(tl.where(is_best_e, e_ids, 256)).to(tl.int32)
        best_s    = tl.sum(tl.where(e_ids == best_idx, s, 0.0))

        topk_idx_arr = tl.where(k_ids == k, best_idx, topk_idx_arr)
        topk_w_arr   = tl.where(k_ids == k, best_s,   topk_w_arr)
        cur_vals     = tl.where(e_ids == best_idx, -1e9, cur_vals)

    # Normalize weights and apply routing scale factor
    w_sum      = tl.sum(topk_w_arr) + 1e-20
    topk_w_arr = topk_w_arr / w_sum * routed_scaling_factor

    tl.store(topk_idx_ptr     + pid * 8 + k_ids, topk_idx_arr)
    tl.store(topk_weights_ptr + pid * 8 + k_ids, topk_w_arr)

    # ── Fused counting: topk_idx_arr is still in registers — no global-memory
    # round-trip needed (cf. the former separate _count_expert_tokens pass). ──
    for k in tl.static_range(8):        # TOP_K = 8
        eid      = tl.sum(tl.where(k_ids == k, topk_idx_arr, 0)).to(tl.int32)
        lid      = eid - local_start
        is_local = (lid >= 0) & (lid < 32)  # E_LOCAL = 32
        tl.atomic_add(expert_counts_ptr + tl.where(is_local, lid, 0), 1, mask=is_local)


@triton.jit
def _init_workspace(
    expert_counts_ptr,    # [E_LOCAL=32]   int32  → zero
    expert_offsets_ptr,   # [E_LOCAL+1=33] int32  → zero
    sorted_weights_ptr,   # [gcap]         f32    → zero
    token_slot_map_ptr,   # [T*TOP_K]      int32  → fill with gcap (sentinel for non-local slots)
    gcap,                 # sorted_weights length; also the sentinel value for token_slot_map
    total_slots,          # T * TOP_K — number of elements in token_slot_map
    BLOCK: tl.constexpr,
):
    """Fuse all per-call init dispatches into one Triton kernel.

    Grid sized by token_slot_map (T*TOP_K = T*8, the largest buffer at 32 KB
    for T=1024).  Since T*8 >= gcap=2T always, this grid also covers
    sorted_weights without any extra CTAs.

    CTA 0 additionally zeroes expert_counts[32] and expert_offsets[33] (trivial,
    128–132 bytes). All CTAs zero sorted_weights[gcap] and fill token_slot_map
    with the gcap sentinel value (non-local slots kept as gcap; _scatter_sorted_tokens
    overwrites valid local slots with the actual workspace position).

    Replaces 4 separate PyTorch dispatches (zero_() × 3 + fill_() × 1).
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)

    # ── CTA 0: zero the two tiny routing arrays ───────────────────────────────
    if pid == 0:
        # expert_counts[32]
        ec_idx = tl.arange(0, 32)
        tl.store(expert_counts_ptr + ec_idx, tl.zeros((32,), dtype=tl.int32))
        # expert_offsets[33]: use next power-of-2 (64) with a mask
        eo_idx = tl.arange(0, 64)
        tl.store(expert_offsets_ptr + eo_idx,
                 tl.zeros((64,), dtype=tl.int32), mask=eo_idx < 33)

    # ── All CTAs: zero sorted_weights[gcap] ──────────────────────────────────
    tl.store(sorted_weights_ptr + off,
             tl.zeros((BLOCK,), dtype=tl.float32), mask=off < gcap)

    # ── All CTAs: fill token_slot_map[T*TOP_K] with gcap sentinel ────────────
    # Sentinel value = gcap; _weighted_output checks slot < gcap to skip these.
    tl.store(token_slot_map_ptr + off,
             tl.zeros((BLOCK,), dtype=tl.int32) + gcap,
             mask=off < total_slots)


@triton.jit
def _prefix_sum(
    counts_ptr,      # [E_LOCAL] int32 — token counts per local expert
    offsets_ptr,     # [E_LOCAL+1] int32 — output: exclusive prefix (offsets[0]=0 pre-zeroed)
    write_ptrs_ptr,  # [E_LOCAL] int32 — output: same exclusive prefix (scatter cursors)
    E: tl.constexpr,
):
    """Single-CTA exclusive prefix sum over E_LOCAL=32 expert counts.

    Replaces three separate PyTorch ops (aten::cumsum + aten::to + aten::copy_)
    with one Triton kernel dispatch.  E=32 fits in a single warp — the entire
    computation lives in registers with no shared-memory traffic.

    On exit:
        offsets[0]   = 0                        (written by caller's zero_())
        offsets[1:E+1] = cumsum(counts)[0:E]    (inclusive → stored at +1 offset)
        write_ptrs[i] = offsets[i]              = exclusive prefix at position i
    """
    idx = tl.arange(0, E)               # [0..E-1], all in one warp for E=32
    c   = tl.load(counts_ptr + idx)     # [E_LOCAL] int32 counts
    cs  = tl.cumsum(c, axis=0)          # inclusive prefix sum in registers
    # offsets[1..E] = cs  (offsets[0]=0 already written by expert_offsets.zero_())
    tl.store(offsets_ptr + 1 + idx, cs.to(tl.int32))
    # write_ptrs[i] = exclusive prefix = cs[i] - c[i]
    tl.store(write_ptrs_ptr + idx, (cs - c).to(tl.int32))


@triton.jit
def _scatter_sorted_tokens(
    topk_idx_ptr,          # [T, 8] int32
    topk_weights_ptr,      # [T, 8] f32
    write_ptrs_ptr,        # [E_LOCAL] int32 — per-expert write cursor (clone of offsets[:-1])
    sorted_token_ids_ptr,  # [gcap] int32 (output)
    sorted_weights_ptr,    # [gcap] f32   (output)
    token_slot_map_ptr,    # [T, 8] int32 (output) — maps (token, k) → workspace slot pos
    local_start, T,
):
    """One CTA per token. Scatter each token into its expert's sorted slice.

    Also fills token_slot_map[pid, k] = pos for local expert assignments so that
    _weighted_output can gather GEMM2 results back to the original token row.
    Non-local slots retain the gcap sentinel from _init_workspace.
    """
    pid = tl.program_id(0)
    if pid >= T:
        return

    for k in tl.static_range(8):        # TOP_K = 8
        eid      = tl.load(topk_idx_ptr     + pid * 8 + k)
        w        = tl.load(topk_weights_ptr + pid * 8 + k)
        lid      = eid - local_start
        is_local = (lid >= 0) & (lid < 32)  # E_LOCAL = 32
        lid_safe = tl.where(is_local, lid, 0)

        # Atomically grab the next write slot for this expert
        pos = tl.atomic_add(write_ptrs_ptr + lid_safe, 1, mask=is_local)
        tl.store(sorted_token_ids_ptr + pos, pid, mask=is_local)
        tl.store(sorted_weights_ptr   + pos, w,   mask=is_local)
        # Store slot position for _weighted_output reverse-gather
        tl.store(token_slot_map_ptr   + pid * 8 + k, pos, mask=is_local)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (destination-passing style: output is the last argument)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _kernel_impl(
    routing_logits:        torch.Tensor,
    routing_bias:          torch.Tensor,
    hidden_states:         torch.Tensor,
    hidden_states_scale:   torch.Tensor,
    gemm1_weights:         torch.Tensor,
    gemm1_weights_scale:   torch.Tensor,
    gemm2_weights:         torch.Tensor,
    gemm2_weights_scale:   torch.Tensor,
    local_expert_offset:   int,
    routed_scaling_factor: float,
    output:                torch.Tensor,
):
    T      = hidden_states.shape[0]
    device = hidden_states.device
    local_start = int(local_expert_offset)

    # ── Get/create persistent workspace (zero per-call allocation overhead) ──
    ws = _ensure_workspace(T, device)
    gcap = ws['gcap']

    # ── 0. Fused workspace init: zero routing arrays + sorted_weights,
    #       fill token_slot_map with gcap sentinel — all in one kernel launch.
    # Grid sized by T*TOP_K (= T*8, largest buffer). Since T*8 >= gcap=2T,
    # the same CTAs also cover sorted_weights (via mask off < gcap).
    # CTA 0 additionally zeroes expert_counts[32] and expert_offsets[33].
    _INIT_BLOCK  = 256
    total_slots  = T * TOP_K
    _init_workspace[(triton.cdiv(total_slots, _INIT_BLOCK),)](
        ws['expert_counts'],
        ws['expert_offsets'],
        ws['sorted_weights'],
        ws['token_slot_map'],
        gcap,
        total_slots,
        BLOCK=_INIT_BLOCK,
    )

    # ── 1+2. Fused routing + counting (Triton, 1 kernel) ────────────────────
    # topk_idx_arr is live in registers at the end of routing, so counting is
    # free — no global-memory write→read round-trip between the two old kernels.
    _routing_kernel[(T,)](
        routing_logits, routing_bias,
        ws['topk_idx'], ws['topk_weights'],
        ws['expert_counts'],
        T, local_start, routed_scaling_factor,
    )

    # ── 3. Triton prefix sum (replaces aten::cumsum + aten::to + aten::copy_) ──
    # Single CTA, 32 threads — computes exclusive prefix sum in registers and
    # writes both expert_offsets[1:] and write_ptrs in one kernel launch.
    # offsets[0] is already 0 from the _init_workspace call above.
    _prefix_sum[(1,)](
        ws['expert_counts'], ws['expert_offsets'], ws['write_ptrs'], E=E_LOCAL,
    )

    # ── 4. Scatter tokens into sorted layout; fill token_slot_map ────────────
    _scatter_sorted_tokens[(T,)](
        ws['topk_idx'], ws['topk_weights'], ws['write_ptrs'],
        ws['sorted_token_ids'], ws['sorted_weights'],
        ws['token_slot_map'],
        local_start, T,
    )

    # ── 5. (removed) — gather fused into GEMM1 via inline token_ids indexing ──

    # ── 6. Grouped GEMM1 with inline gather → [gcap, 2*I] f32 ───────────────
    # hidden_states [T, H] and hidden_states_scale [H//128, T] are passed raw;
    # GEMM1 resolves workspace slot → token row via sorted_token_ids internally,
    # eliminating the two aten::index + contiguous() calls from v16.
    N1 = 2 * I
    # Grid M-dim: ceil(gcap/BM) + E_LOCAL.
    # cum_m <= total_tokens/BM + E_LOCAL (each expert adds at most 1 partial tile),
    # so grid_m >= cum_m always (given total_tokens <= gcap).
    grid1 = lambda meta: (
        triton.cdiv(gcap, meta['BLOCK_M']) + E_LOCAL,
        triton.cdiv(N1, meta['BLOCK_N']),
    )
    _grouped_gemm1_swiglu[grid1](
        ws['sorted_token_ids'],                         # token_ids (new first arg)
        hidden_states,          hidden_states_scale,    # A raw [T,H], SA [H//128,T]
        gemm1_weights,          gemm1_weights_scale,
        ws['gemm1_out'],
        ws['expert_offsets'],
        N1, H, gcap,
        hidden_states.stride(0),        hidden_states.stride(1),
        hidden_states_scale.stride(0),  hidden_states_scale.stride(1),
        gemm1_weights.stride(0),        gemm1_weights.stride(1),  gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0),  gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        ws['gemm1_out'].stride(0),      ws['gemm1_out'].stride(1),
    )

    # ── 7. SwiGLU + per-row × per-128-K-block FP16 quantization ──────────────
    _swiglu_to_fp16_scaled[
        (triton.cdiv(gcap, 32), triton.cdiv(I, 128))
    ](
        ws['gemm1_out'], ws['swiglu_fp16'], ws['swiglu_scale_a'],
        gcap,
        ws['swiglu_scale_a'].stride(0), ws['swiglu_scale_a'].stride(1),
        I=I, BLOCK_M=32, BLOCK_I=128,
    )

    # ── 8. Grouped GEMM2: FP16×FP8→FP32, plain store to gemm2_out [gcap, H] ──
    # Each workspace slot rm gets exactly one tl.store — no atomic contention.
    grid2 = lambda meta: (
        triton.cdiv(gcap, meta['BLOCK_M']) + E_LOCAL,
        triton.cdiv(H, meta['BLOCK_N']),
    )
    _grouped_gemm2[grid2](
        ws['swiglu_fp16'],              ws['swiglu_scale_a'],
        gemm2_weights,                  gemm2_weights_scale,
        ws['gemm2_out'],
        ws['expert_offsets'],
        H, I, gcap,
        ws['swiglu_fp16'].stride(0),    ws['swiglu_fp16'].stride(1),
        ws['swiglu_scale_a'].stride(0), ws['swiglu_scale_a'].stride(1),
        gemm2_weights.stride(0),        gemm2_weights.stride(1), gemm2_weights.stride(2),
        gemm2_weights_scale.stride(0),  gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
        ws['gemm2_out'].stride(0),      ws['gemm2_out'].stride(1),
    )

    # ── 9. Weighted gather: gemm2_out[slot,:] * weight → output [T, H] bf16 ──
    # Each output token CTA iterates over its 8 expert slots, accumulates fp32,
    # stores bf16. No atomics needed — each output token row is owned by one CTA.
    _weighted_output[(T, triton.cdiv(H, 128))](
        ws['gemm2_out'], ws['sorted_weights'], ws['token_slot_map'],
        output, T, H, gcap, BLOCK_N=128,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CUDA Graph wrapper — eliminates ~95µs of inter-kernel host launch latency
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def kernel(
    routing_logits:        torch.Tensor,
    routing_bias:          torch.Tensor,
    hidden_states:         torch.Tensor,
    hidden_states_scale:   torch.Tensor,
    gemm1_weights:         torch.Tensor,
    gemm1_weights_scale:   torch.Tensor,
    gemm2_weights:         torch.Tensor,
    gemm2_weights_scale:   torch.Tensor,
    local_expert_offset:   int,
    routed_scaling_factor: float,
    output:                torch.Tensor,
):
    """CUDA-Graph-cached entry point.

    flashinfer-bench reuses the same input tensors across the 100 timed
    iterations of a trial (verified against bench_gpu_time_with_cupti's
    closed-over args), so the graph keyed by (T, all data_ptrs, scalars) hits
    on iter 2..100 of every trial.  A new trial allocates fresh tensors → key
    miss → re-capture once, then 99 hits.

    Capture-time considerations:
      - Triton @autotune trials would corrupt the graph (they launch each
        config repeatedly during capture).  We do an eager warmup call before
        capture to populate the autotune cache, so capture-time launches go
        through the cache-hit fast path.
      - Capture must run on a non-default stream (CUDA requirement).
      - The persistent workspace tensors (_ws_cache) are bound into the graph
        by data_ptr at capture time; they must live for the graph's lifetime.
        _ws_cache is module-global so this is automatic.
    """
    T = hidden_states.shape[0]

    # Cache key: T + data_ptrs of every tensor + scalar args.  Pointer change
    # (e.g. across trials) invalidates the cached graph.
    key = (
        T,
        routing_logits.data_ptr(),    routing_bias.data_ptr(),
        hidden_states.data_ptr(),     hidden_states_scale.data_ptr(),
        gemm1_weights.data_ptr(),     gemm1_weights_scale.data_ptr(),
        gemm2_weights.data_ptr(),     gemm2_weights_scale.data_ptr(),
        output.data_ptr(),
        int(local_expert_offset),
        float(routed_scaling_factor),
        str(hidden_states.device),
    )

    entry = _graph_cache.get(key)
    if entry is not None:
        entry.replay()
        return

    # ── Cache miss: warmup (populate autotune cache) then capture ───────────
    # One eager call so all @triton.autotune-decorated kernels resolve their
    # best config and write it to the autotune dict.  Without this, capture
    # would record the autotune trial-launch sequence.
    _kernel_impl(
        routing_logits, routing_bias,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor,
        output,
    )

    # Capture on a side stream as required by CUDA Graph API.
    g = torch.cuda.CUDAGraph()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        with torch.cuda.graph(g):
            _kernel_impl(
                routing_logits, routing_bias,
                hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale,
                local_expert_offset, routed_scaling_factor,
                output,
            )
    torch.cuda.current_stream().wait_stream(side_stream)

    _graph_cache[key] = g
