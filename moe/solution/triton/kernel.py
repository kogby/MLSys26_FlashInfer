"""
Triton FP8 Fused MoE kernel — v16

v16 replaces index_add_ scatter with a token-parallel gather kernel.

v15 (previous):
- FP16 GEMM2 with overflow-safe A-side scaling (per-row × per-128-K-block).
- Zero CPU syncs via persistent workspace (v12).

v16 (this version):
- Step 9 was 4 ops (mul + zeros + index_add_ + copy_) costing ~343µs for
  T=2048, dominated by index_add_: a scatter with random writes that breaks
  memory coalescing and requires atomics for duplicate token IDs.
- `_scatter_sorted_tokens` now also fills `token_slot_map[T, TOP_K]`: for
  each top-k assignment that lands on a local expert, stores the workspace
  slot index. Non-local entries retain the gcap sentinel.
- New `_weighted_output` kernel: grid=(T, H//128), one CTA per (token, column
  block). Iterates k=0..7, reads gemm2_out[slot, :] and sorted_weights[slot],
  accumulates in float32, writes bfloat16 directly to output. No intermediates,
  no atomics, coalesced writes. Estimated memory traffic: 557MB → 87MB.

Pipeline (all GPU, no CPU sync):
  1. _routing_kernel         — Triton; [T,8] indices + weights
  2. _count_expert_tokens    — Triton; expert token counts
  3. GPU cumsum (no CPU sync) — expert_offsets [E_LOCAL+1]
  4. _scatter_sorted_tokens  — Triton; sorted_token_ids, sorted_weights,
                               token_slot_map
  5. _grouped_gemm1_swiglu   — FP8×FP8→float32 (FP8 tensor cores)
  6. _swiglu_to_fp16_scaled  — silu(up)*gate FP32; per-row/per-128 scale → FP16
  7. _grouped_gemm2          — FP16×FP16→float32 with two-sided block scaling
  8. _weighted_output        — gather gemm2_out via slot_map → bf16 output
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
        # token_slot_map[t, k] = workspace slot for token t's k-th top-k expert,
        # or gcap (sentinel) if that assignment is not a local expert.
        'token_slot_map':   torch.full((T, TOP_K), gcap, dtype=torch.int32, device=device),
        'gemm1_out':        torch.zeros((gcap, 2 * I), dtype=torch.float32, device=device),
        'swiglu_fp16':      torch.empty((gcap, I),     dtype=torch.float16, device=device),
        'swiglu_scale_a':   torch.empty((I // QUANT_BLOCK, gcap), dtype=torch.float32, device=device),
        'gemm2_out':        torch.zeros((gcap, H),     dtype=torch.float32, device=device),
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
    # A: sorted hidden states [total_tokens, K] FP8
    A_ptr, SA_ptr,
    # B: W1 weights [E_LOCAL, 2*I, K] FP8  (N = 2*I)
    B_ptr, SB_ptr,
    # C: SwiGLU output [total_tokens, I] f32
    C_ptr,
    # expert token ranges
    expert_offsets_ptr,       # [E_LOCAL+1] int32
    # dims
    N, K,                     # N = 2*I, K = H
    total_tokens,             # autotune key — total rows across all experts
    # strides for A [total_tokens, K]
    stride_am, stride_ak,
    # strides for SA [K//128, total_tokens]
    stride_sA_kb, stride_sA_m,
    # strides for B [E_LOCAL, 2*I, K]
    stride_be, stride_bn, stride_bk,
    # strides for SB [E_LOCAL, 2*I//128, K//128]
    stride_sB_e, stride_sB_nb, stride_sB_kb,
    # strides for C [total_tokens, I]
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(K // BLOCK_K):
        rk = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm_local[:, None] < M_e) & (rk[None, :] < K), other=0.0,
        )
        sa = tl.load(
            SA_ptr + kb * stride_sA_kb + rm * stride_sA_m,
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
    B_ptr, SB_ptr,            # B: FP8 weights, SB: per-128-block scales
    C_ptr,                    # [total_tokens, N] FP32  (output)
    expert_offsets_ptr,
    N, K,
    total_tokens,
    stride_am, stride_ak,
    stride_sA_kb, stride_sA_m,
    stride_be, stride_bn, stride_bk,
    stride_sB_e, stride_sB_nb, stride_sB_kb,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(K // BLOCK_K):
        rk = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        # A is FP16 (per-row × per-128-K-block scaled SwiGLU output).
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm_local[:, None] < M_e) & (rk[None, :] < K), other=0.0,
        )
        sa = tl.load(
            SA_ptr + kb * stride_sA_kb + rm * stride_sA_m,
            mask=rm_local < M_e, other=0.0,
        )

        # B is FP8 → FP16 directly (lossless: FP8 max=448 fits, 3-bit
        # mantissa fits in 10-bit FP16). Per-128 SB factored outside dot.
        b = tl.load(
            B_ptr + pid_e * stride_be + rn[:, None] * stride_bn + rk[None, :] * stride_bk,
            mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0,
        )
        sb     = tl.load(SB_ptr + pid_e * stride_sB_e + pid_n * stride_sB_nb + kb * stride_sB_kb)
        b_fp16 = b.to(tl.float16)

        # FP16 × FP16 → FP32 tensor core (HMMA on Hopper, similar on Blackwell).
        # Both block scales are constant across the reduction → applied to acc.
        partial = tl.dot(a, tl.trans(b_fp16), out_dtype=tl.float32)
        acc = acc + partial * sa[:, None] * sb

    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm_local[:, None] < M_e) & (rn[None, :] < N),
    )


# ─────────────────────────────────────────────────────────────────────────────
# v11 routing kernels
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _routing_kernel(
    logits_ptr,           # [T, 256] f32
    bias_ptr,             # [256] f32
    topk_idx_ptr,         # [T, 8] int32   (output)
    topk_weights_ptr,     # [T, 8] f32     (output)
    T, routed_scaling_factor,
):
    """One CTA per token. Fuses sigmoid+bias, group scoring, top-K selection."""
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


@triton.jit
def _count_expert_tokens(
    topk_idx_ptr,         # [T, 8] int32
    expert_counts_ptr,    # [E_LOCAL] int32 (output, zeroed before call)
    local_start, T,
):
    """One CTA per token. Atomically increments count for each local expert."""
    pid = tl.program_id(0)
    if pid >= T:
        return

    for k in tl.static_range(8):        # TOP_K = 8
        eid      = tl.load(topk_idx_ptr + pid * 8 + k)
        lid      = eid - local_start
        is_local = (lid >= 0) & (lid < 32)  # E_LOCAL = 32
        tl.atomic_add(expert_counts_ptr + tl.where(is_local, lid, 0), 1, mask=is_local)


@triton.jit
def _scatter_sorted_tokens(
    topk_idx_ptr,          # [T, 8] int32
    topk_weights_ptr,      # [T, 8] f32
    write_ptrs_ptr,        # [E_LOCAL] int32 — per-expert write cursor (clone of offsets[:-1])
    sorted_token_ids_ptr,  # [total_tokens] int32 (output)
    sorted_weights_ptr,    # [total_tokens] f32   (output)
    slot_map_ptr,          # [T, TOP_K] int32 (output) — workspace slot per (token, top_k_k)
    local_start, T,
):
    """One CTA per token. Scatter each token into its expert's sorted slice.
    Also fills slot_map[pid, k] = workspace slot for local assignments so the
    output gather kernel can find each token's contributions without index_add_.
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
        # Record which workspace slot this (token, top_k_k) maps to.
        # Non-local k entries retain the gcap sentinel written during workspace init.
        tl.store(slot_map_ptr + pid * 8 + k, pos, mask=is_local)


@triton.jit
def _weighted_output(
    gemm2_out_ptr,       # [gcap, H] f32  — GEMM2 output
    sorted_weights_ptr,  # [gcap] f32     — per-slot routing weight
    slot_map_ptr,        # [T, TOP_K] int32 — workspace slot per (token, k); gcap = sentinel
    out_ptr,             # [T, H] bf16    — final output (written directly)
    T, H, gcap,
    BLOCK_N: tl.constexpr,
):
    """Token-parallel gather: one CTA per (output token, column block).

    For each output token, reads up to TOP_K=8 rows from gemm2_out (identified
    via slot_map), multiplies each by its routing weight, accumulates in float32,
    then stores as bfloat16. Writes are coalesced (consecutive threads write
    consecutive columns of the same token row). No atomics needed.
    """
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    rn     = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < H

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in tl.static_range(8):          # TOP_K = 8
        slot     = tl.load(slot_map_ptr + pid_t * 8 + k)
        is_valid = slot < gcap             # gcap is sentinel for "not a local expert"
        slot_s   = tl.where(is_valid, slot, 0)

        w   = tl.load(sorted_weights_ptr + slot_s, mask=is_valid, other=0.0)
        row = tl.load(gemm2_out_ptr + slot_s * H + rn,
                      mask=is_valid & mask_n, other=0.0)
        acc = acc + row * w

    tl.store(out_ptr + pid_t * H + rn, acc.to(tl.bfloat16), mask=mask_n)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (destination-passing style: output is the last argument)
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
    T      = hidden_states.shape[0]
    device = hidden_states.device
    local_start = int(local_expert_offset)

    # ── Get/create persistent workspace (zero per-call allocation overhead) ──
    ws = _ensure_workspace(T, device)
    gcap = ws['gcap']

    # ── Re-init mutable buffers (all GPU ops, near-zero latency) ─────────────
    ws['expert_counts'].zero_()
    ws['expert_offsets'].zero_()
    ws['sorted_weights'].zero_()        # KEY INVARIANT: invalid rows → 0 weight → 0 output contribution
    ws['token_slot_map'].fill_(gcap)    # reset sentinel; scatter will overwrite local assignments

    # ── 1. Fused routing (Triton, 1 kernel) ──────────────────────────────────
    _routing_kernel[(T,)](
        routing_logits, routing_bias,
        ws['topk_idx'], ws['topk_weights'],
        T, routed_scaling_factor,
    )

    # ── 2. Count tokens per local expert (Triton, 1 kernel) ──────────────────
    _count_expert_tokens[(T,)](
        ws['topk_idx'], ws['expert_counts'], local_start, T,
    )

    # ── 3. GPU prefix sum (no CPU sync) ──────────────────────────────────────
    ws['expert_offsets'][1:] = ws['expert_counts'].cumsum(0).to(torch.int32)
    ws['write_ptrs'].copy_(ws['expert_offsets'][:-1])

    # ── 4. Scatter tokens into sorted layout (Triton, 1 kernel) ──────────────
    _scatter_sorted_tokens[(T,)](
        ws['topk_idx'], ws['topk_weights'], ws['write_ptrs'],
        ws['sorted_token_ids'], ws['sorted_weights'],
        ws['token_slot_map'],
        local_start, T,
    )

    # ── 5. Gather sorted hidden states (gcap rows; invalid rows read row 0) ──
    sorted_A       = hidden_states[ws['sorted_token_ids']].contiguous()
    sorted_A_scale = hidden_states_scale[:, ws['sorted_token_ids']].contiguous()

    # ── 6. Grouped GEMM1 → [gcap, 2*I] f32 ──────────────────────────────────
    N1 = 2 * I
    # Grid M-dim: ceil(gcap/BM) + E_LOCAL.
    # cum_m <= total_tokens/BM + E_LOCAL (each expert adds at most 1 partial tile),
    # so grid_m >= cum_m always (given total_tokens <= gcap).
    grid1 = lambda meta: (
        triton.cdiv(gcap, meta['BLOCK_M']) + E_LOCAL,
        triton.cdiv(N1, meta['BLOCK_N']),
    )
    _grouped_gemm1_swiglu[grid1](
        sorted_A,       sorted_A_scale,
        gemm1_weights,  gemm1_weights_scale,
        ws['gemm1_out'],
        ws['expert_offsets'],
        N1, H, gcap,
        sorted_A.stride(0),       sorted_A.stride(1),
        sorted_A_scale.stride(0), sorted_A_scale.stride(1),
        gemm1_weights.stride(0),  gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        ws['gemm1_out'].stride(0), ws['gemm1_out'].stride(1),
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

    # ── 7. Grouped GEMM2 → [gcap, H] ─────────────────────────────────────────
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

    # ── 8. Token-parallel gather → bfloat16 output ───────────────────────────
    # Each CTA handles one (output token, column block). Reads up to 8 gemm2_out
    # rows via token_slot_map, multiplies by sorted_weights, writes bf16 directly.
    # No intermediates, no atomics, coalesced writes. ~557MB → ~87MB traffic.
    _weighted_output[(T, triton.cdiv(H, 128))](
        ws['gemm2_out'], ws['sorted_weights'], ws['token_slot_map'],
        output, T, H, gcap, BLOCK_N=128,
    )
