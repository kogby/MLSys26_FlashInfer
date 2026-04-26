"""
Triton FP8 Fused MoE kernel — v8

v8 vs v7:
- Persistent 2D tile scheduling for both grouped GEMMs.
  Old: 3D grid (ceil(max_tokens/BM), N_TILES, E_LOCAL) — wastes CTAs for under-loaded experts.
  New: 2D grid (total_valid_m_tiles, N_TILES) — only launches tiles for experts with tokens.
  In-kernel decode: 32-iteration static loop maps global m-tile → (expert, local m-tile).
- Eliminates up to 75% wasted CTAs on imbalanced workloads; all SMs stay busy.

Pipeline:
  1. _compute_routing      — PyTorch routing, returns [T,8] indices + dense [T,256] weights
  2. _build_expert_map     — sort (token,expert) pairs → sorted_ids, expert_offsets
  3. _grouped_gemm1_swiglu — FP8×FP8→float32, writes [total_tokens, 2*I]
  4. _swiglu_inplace       — silu(up)*gate in-place → [total_tokens, I] in first I cols
  5. _grouped_gemm2        — float32×FP8→float32, reads first I cols via stride trick
  6. index_add_            — weighted scatter into output
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
# Autotune configs — same search space as v3, BLOCK_M is the key variable
# ─────────────────────────────────────────────────────────────────────────────
_GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': 128, 'BLOCK_K': 128},
                  num_warps=NW, num_stages=NS)
    for BM in [16, 32, 64, 128]
    for NW in [4, 8]
    for NS in [3, 4, 5]
]

# GEMM1 has a single accumulator (SwiGLU is handled by a separate _swiglu_inplace
# pass, not two in-register accumulators), so full BLOCK_M range is safe.
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
    # strides for B [E_LOCAL, 2*I, K]  (B is indexed as [expert, n, k])
    stride_be, stride_bn, stride_bk,
    # strides for SB [E_LOCAL, 2*I//128, K//128]
    stride_sB_e, stride_sB_nb, stride_sB_kb,
    # strides for C [total_tokens, I]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m_global = tl.program_id(0)   # global m-tile index across all experts
    pid_n        = tl.program_id(1)   # tile index over 2*I output cols

    # Decode pid_m_global → (expert id, local m-tile within that expert).
    # Linear scan over E_LOCAL=32 experts; unrolled at compile time via static_range.
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

    # Token range for the decoded expert
    e_start = tl.load(expert_offsets_ptr + pid_e).to(tl.int32)
    e_end   = tl.load(expert_offsets_ptr + pid_e + 1).to(tl.int32)
    M_e     = e_end - e_start

    # Row indices within this expert's slice
    rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rm       = e_start + rm_local   # absolute row in the sorted buffer
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
        a = a.to(tl.float32) * sa[:, None]

        b = tl.load(
            B_ptr + pid_e * stride_be + rn[:, None] * stride_bn + rk[None, :] * stride_bk,
            mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0,
        )
        sb = tl.load(SB_ptr + pid_e * stride_sB_e + pid_n * stride_sB_nb + kb * stride_sB_kb)
        b  = b.to(tl.float32) * sb

        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

    # Write combined [gate||up] intermediate; _swiglu_inplace fuses SwiGLU in a
    # second pass so each CTA here carries only one accumulator (no register pressure).
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm_local[:, None] < M_e) & (rn[None, :] < N),
    )


@triton.jit
def _swiglu_inplace(
    X_ptr,          # [total_tokens, 2*I] f32  — GEMM1 output, gate||up
    total_tokens,
    I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Read gate and up from X, write silu(up)*gate back into the first I cols."""
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ri = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    gate = tl.load(X_ptr + rm[:, None] * (2 * I) + ri[None, :],
                   mask=(rm[:, None] < total_tokens) & (ri[None, :] < I), other=0.0)
    up   = tl.load(X_ptr + rm[:, None] * (2 * I) + (ri[None, :] + I),
                   mask=(rm[:, None] < total_tokens) & (ri[None, :] < I), other=0.0)

    z = tl.sigmoid(up) * up * gate   # silu(up) * gate

    tl.store(X_ptr + rm[:, None] * (2 * I) + ri[None, :],
             z,
             mask=(rm[:, None] < total_tokens) & (ri[None, :] < I))


# ─────────────────────────────────────────────────────────────────────────────
# Grouped GEMM2
#
# A: SwiGLU output [total_tokens, I] f32   (first I cols of the GEMM1 buffer)
# B: W2 weights   [E_LOCAL, H, I]    FP8
# SB:             [E_LOCAL, H//128, I//128]
# C: output       [total_tokens, H]  f32
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(configs=_GEMM_CONFIGS, key=['N', 'K', 'total_tokens'])
@triton.jit
def _grouped_gemm2(
    A_ptr,
    B_ptr, SB_ptr,
    C_ptr,
    expert_offsets_ptr,
    N, K,           # N = H = 7168, K = I = 2048
    total_tokens,   # autotune key — total rows across all experts
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_sB_e, stride_sB_nb, stride_sB_kb,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m_global = tl.program_id(0)
    pid_n        = tl.program_id(1)

    # Decode pid_m_global → (expert id, local m-tile within that expert).
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
        ).to(tl.float32)

        b = tl.load(
            B_ptr + pid_e * stride_be + rn[:, None] * stride_bn + rk[None, :] * stride_bk,
            mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0,
        )
        sb = tl.load(SB_ptr + pid_e * stride_sB_e + pid_n * stride_sB_nb + kb * stride_sB_kb)
        b  = b.to(tl.float32) * sb

        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm_local[:, None] < M_e) & (rn[None, :] < N),
    )


# Weighted scatter-add is done in PyTorch via index_add_ (handles repeated
# indices atomically and is better optimized than a Triton atomic kernel for
# this pattern — avoids launching total_tokens × H/BLOCK_H tiny CTAs).


# ─────────────────────────────────────────────────────────────────────────────
# Routing (pure PyTorch — same as v3)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_routing(routing_logits, routing_bias, routed_scaling_factor):
    T = routing_logits.shape[0]

    s      = torch.sigmoid(routing_logits)
    s_bias = s + routing_bias.to(torch.float32)

    s_grouped    = s_bias.view(T, N_GROUP, E_GLOBAL // N_GROUP)
    top2, _      = torch.topk(s_grouped, k=2, dim=2, sorted=False)
    group_scores = top2.sum(dim=2)

    _, grp_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, sorted=False)
    grp_mask   = torch.zeros_like(group_scores)
    grp_mask.scatter_(1, grp_idx, 1.0)
    score_mask = (
        grp_mask.unsqueeze(2)
        .expand(T, N_GROUP, E_GLOBAL // N_GROUP)
        .reshape(T, E_GLOBAL)
    )

    masked      = s_bias.masked_fill(score_mask == 0, float("-inf"))
    _, topk_idx = torch.topk(masked, k=TOP_K, dim=1, sorted=False)

    weight_mask = torch.zeros_like(s)
    weight_mask.scatter_(1, topk_idx, 1.0)
    weights = s * weight_mask
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-20)
    weights = weights * routed_scaling_factor

    return topk_idx, weights


# ─────────────────────────────────────────────────────────────────────────────
# Build expert map: sort (token, expert) pairs → contiguous layout
#
# Returns:
#   sorted_token_ids  [total_tokens]   int32 — original token index for each row
#   sorted_weights    [total_tokens]   f32   — routing weight for that (token, expert) pair
#   expert_offsets    [E_LOCAL+1]      int32 — start of each expert's slice
#   total_tokens      int              — sum of all expert token counts
# ─────────────────────────────────────────────────────────────────────────────

def _build_expert_map(topk_idx, weights, local_start, device):
    T = topk_idx.shape[0]

    expert_ids = topk_idx.reshape(-1)                        # [T*TOP_K]
    token_ids  = torch.arange(T, device=device).repeat_interleave(TOP_K)  # [T*TOP_K]

    local_mask = (expert_ids >= local_start) & (expert_ids < local_start + E_LOCAL)
    local_expert_ids = expert_ids[local_mask] - local_start
    local_token_ids  = token_ids[local_mask]

    global_expert_ids = expert_ids[local_mask]
    local_weights = weights[local_token_ids, global_expert_ids]

    sort_order        = torch.argsort(local_expert_ids, stable=True)
    sorted_expert_ids = local_expert_ids[sort_order]
    sorted_token_ids  = local_token_ids[sort_order]
    sorted_weights    = local_weights[sort_order]

    counts = torch.bincount(sorted_expert_ids, minlength=E_LOCAL)
    expert_offsets = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = counts.cumsum(0).to(torch.int32)

    total_tokens     = int(expert_offsets[E_LOCAL].item())
    sorted_token_ids = sorted_token_ids.to(torch.int32)
    return sorted_token_ids, sorted_weights, expert_offsets, total_tokens


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

    # ── 1. Routing ────────────────────────────────────────────────────────────
    topk_idx, weights = _compute_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )

    # ── 2. Build expert map ───────────────────────────────────────────────────
    sorted_token_ids, sorted_weights, expert_offsets, total_tokens = \
        _build_expert_map(topk_idx, weights, local_start, device)

    if total_tokens == 0:
        output.zero_()
        return

    # ── 3. Gather sorted hidden states ────────────────────────────────────────
    # sorted_A: [total_tokens, H] FP8 — each row is hidden_states[sorted_token_ids[i]]
    sorted_A       = hidden_states[sorted_token_ids].contiguous()
    # hidden_states_scale: [H//128, T] → gather columns → [H//128, total_tokens]
    sorted_A_scale = hidden_states_scale[:, sorted_token_ids].contiguous()

    # ── 4. Grouped GEMM1 → [total_tokens, 2*I] f32 ───────────────────────────
    N1 = 2 * I  # = 4096
    gemm1_out = torch.empty((total_tokens, N1), dtype=torch.float32, device=device)

    # Persistent 2D tiling: only launch M-tiles for experts that actually have tokens.
    # token_counts_cpu is a list of 32 ints; the CPU transfer is one small sync point
    # (same cost as the original .max().item() call) but eliminates up to 75% of
    # wasted CTAs for imbalanced workloads.
    token_counts_cpu = (expert_offsets[1:] - expert_offsets[:-1]).cpu().tolist()

    grid1 = lambda meta: (
        sum((int(tc) + meta['BLOCK_M'] - 1) // meta['BLOCK_M'] for tc in token_counts_cpu),
        triton.cdiv(N1, meta['BLOCK_N']),
    )
    _grouped_gemm1_swiglu[grid1](
        sorted_A,       sorted_A_scale,
        gemm1_weights,  gemm1_weights_scale,
        gemm1_out,
        expert_offsets,
        N1, H, total_tokens,
        sorted_A.stride(0),       sorted_A.stride(1),
        sorted_A_scale.stride(0), sorted_A_scale.stride(1),
        gemm1_weights.stride(0),  gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        gemm1_out.stride(0),      gemm1_out.stride(1),
    )

    # ── 5. SwiGLU in-place: gemm1_out[:,I:2I] → silu(up)*gate, stored in [:, :I] ──
    BLOCK_I = 128
    SWIGLU_BLOCK_M = 32
    _swiglu_inplace[(triton.cdiv(total_tokens, SWIGLU_BLOCK_M), triton.cdiv(I, BLOCK_I))](
        gemm1_out,
        total_tokens,
        I, SWIGLU_BLOCK_M, BLOCK_I,
    )
    # Now gemm1_out[:, :I] holds the SwiGLU result; rest is garbage.
    # Pass gemm1_out directly to GEMM2 with stride (2*I, 1) so we read only
    # the first I cols without an extra copy.

    # ── 6. Grouped GEMM2 → [total_tokens, H] f32 ─────────────────────────────
    gemm2_out = torch.empty((total_tokens, H), dtype=torch.float32, device=device)

    grid2 = lambda meta: (
        sum((int(tc) + meta['BLOCK_M'] - 1) // meta['BLOCK_M'] for tc in token_counts_cpu),
        triton.cdiv(H, meta['BLOCK_N']),
    )
    _grouped_gemm2[grid2](
        gemm1_out,               # base ptr; stride_am=2*I reads only first I cols
        gemm2_weights,  gemm2_weights_scale,
        gemm2_out,
        expert_offsets,
        H, I, total_tokens,
        gemm1_out.stride(0),     gemm1_out.stride(1),   # (2*I, 1) — correct K stride
        gemm2_weights.stride(0),  gemm2_weights.stride(1), gemm2_weights.stride(2),
        gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
        gemm2_out.stride(0),      gemm2_out.stride(1),
    )

    # ── 7. Weighted scatter-add → out_f32 [T, H] ─────────────────────────────
    out_f32 = torch.zeros((T, H), dtype=torch.float32, device=device)
    weighted = gemm2_out * sorted_weights.unsqueeze(1)   # [total_tokens, H]
    out_f32.index_add_(0, sorted_token_ids.long(), weighted)

    output.copy_(out_f32)
