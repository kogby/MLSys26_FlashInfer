"""
Triton FP8 Fused MoE kernel — v5.

Key change vs v3: replace the 32-iteration Python expert loop with a single
grouped (batched) GEMM launch per stage.

Pipeline:
  1. _compute_routing      — same pure-PyTorch routing as v3, returns [T,8]
  2. _build_expert_map     — sort (token, expert) pairs → sorted_ids, expert_offsets
  3. _grouped_gemm1_swiglu — one Triton launch covering all experts:
                             FP8*FP8 GEMM1 + SwiGLU fused in epilogue → [TK, I] f32
  4. _grouped_gemm2        — one Triton launch covering all experts:
                             f32*FP8 GEMM2 → [TK, H] f32
  5. _scatter_add          — weighted scatter-accumulate into output

Grid shape: (cdiv(max_tokens_per_expert, BLOCK_M), cdiv(N, BLOCK_N), num_active_experts)
Each CTA reads expert_offsets[pid_e]..expert_offsets[pid_e+1] to find its M-range.
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
    for NS in [3, 4]
]


# ─────────────────────────────────────────────────────────────────────────────
# Grouped GEMM1 + SwiGLU fused
#
# Processes all active experts in one launch via a 3-D grid:
#   pid_m = tl.program_id(0)  — token tile within this expert
#   pid_n = tl.program_id(1)  — output-feature tile
#   pid_e = tl.program_id(2)  — expert index (into sorted token buffer)
#
# A (hidden states, FP8): all experts' tokens packed row-by-row in sorted order
#   shape: [total_tokens, K]   where total_tokens = sum of tokens per expert
# SA (A scales):            [K//128, total_tokens]
# B  (W1 weights, FP8):     [E_LOCAL, 2*I, K]
# SB (W1 scales):           [E_LOCAL, 2*I//128, K//128]
# C  (SwiGLU output, f32):  [total_tokens, I]
# expert_offsets:           [E_LOCAL+1]  — start row in A/C for each expert
#
# The fused SwiGLU: at end of K-loop, acc holds [BLOCK_M, 2*I] conceptually
# split into gate (first I cols) and up (second I cols).  We carry two separate
# [BLOCK_M, BLOCK_N] accumulators and write silu(up)*gate directly.
# Because 2*I cols need two separate n-tiles this is handled by pid_n:
#   pid_n < I//BLOCK_N       → gate tile, accumulates into acc_gate
#   pid_n >= I//BLOCK_N      → up   tile, accumulates into acc_up
# Then the two halves are combined in a final fused pass.
#
# Implementation note: carrying two BLOCK_M×BLOCK_N accumulators doubles
# register pressure.  We therefore cap BLOCK_M at 64 for this kernel.
# ─────────────────────────────────────────────────────────────────────────────

# Separate configs with smaller BLOCK_M to avoid register spill from two accumulators
_GEMM1_CONFIGS = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': 128, 'BLOCK_K': 128},
                  num_warps=NW, num_stages=NS)
    for BM in [16, 32, 64]
    for NW in [4, 8]
    for NS in [3, 4]
]


@triton.autotune(configs=_GEMM1_CONFIGS, key=['N', 'K', 'max_tokens'])
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
    max_tokens,               # padded grid size in M dimension
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)   # tile index over 2*I output cols
    pid_e = tl.program_id(2)   # expert index

    # Token range for this expert
    e_start = tl.load(expert_offsets_ptr + pid_e)
    e_end   = tl.load(expert_offsets_ptr + pid_e + 1)
    M_e = e_end - e_start       # number of tokens for this expert

    # Row indices within this expert's slice
    rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rm = e_start + rm_local     # absolute row in the sorted buffer
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

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

    # SwiGLU fused epilogue:
    # pid_n < I//BLOCK_N  → gate half  (cols 0..I-1  of W1 output)
    # pid_n >= I//BLOCK_N → up   half  (cols I..2I-1 of W1 output)
    # We need acc_gate and acc_up for the same (pid_m, pid_e) rows but
    # different col tiles.  Because a single CTA only covers one pid_n tile,
    # we cannot fuse across the gate/up split within one CTA directly.
    # Instead we write a combined intermediate of shape [total_tokens, 2*I]
    # and do SwiGLU in a second light pass (_swiglu_inplace).
    # This avoids the two-accumulator register-pressure problem entirely.
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

@triton.autotune(configs=_GEMM_CONFIGS, key=['N', 'K', 'max_tokens'])
@triton.jit
def _grouped_gemm2(
    A_ptr,
    B_ptr, SB_ptr,
    C_ptr,
    expert_offsets_ptr,
    N, K,           # N = H = 7168, K = I = 2048
    max_tokens,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_sB_e, stride_sB_nb, stride_sB_kb,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)

    e_start = tl.load(expert_offsets_ptr + pid_e)
    e_end   = tl.load(expert_offsets_ptr + pid_e + 1)
    M_e = e_end - e_start

    rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rm = e_start + rm_local
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

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

    # For each of the T*TOP_K (token, expert) assignments, check which fall
    # into local experts [local_start, local_start+E_LOCAL)
    expert_ids = topk_idx.reshape(-1)                        # [T*TOP_K]
    token_ids  = torch.arange(T, device=device).repeat_interleave(TOP_K)  # [T*TOP_K]

    # Filter to local experts only
    local_mask = (expert_ids >= local_start) & (expert_ids < local_start + E_LOCAL)
    local_expert_ids = expert_ids[local_mask] - local_start  # [total_tokens] in [0, E_LOCAL)
    local_token_ids  = token_ids[local_mask]                 # [total_tokens]

    # Gather corresponding weights — weights is [T, E_GLOBAL]
    # For each surviving pair, weight = weights[token_id, global_expert_id]
    global_expert_ids = expert_ids[local_mask]
    local_weights = weights[local_token_ids, global_expert_ids]  # [total_tokens]

    # Sort by local expert id so tokens for each expert are contiguous
    sort_order        = torch.argsort(local_expert_ids, stable=True)
    sorted_expert_ids = local_expert_ids[sort_order]
    sorted_token_ids  = local_token_ids[sort_order]
    sorted_weights    = local_weights[sort_order]

    # Compute expert_offsets via bincount
    counts = torch.bincount(sorted_expert_ids, minlength=E_LOCAL)  # [E_LOCAL]
    expert_offsets = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = counts.cumsum(0).to(torch.int32)

    total_tokens = int(expert_offsets[E_LOCAL].item())
    # int32 for use as pointer offsets in Triton kernels
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

    max_tokens_per_expert = int((expert_offsets[1:] - expert_offsets[:-1]).max().item())
    # Pad to at least BLOCK_M=16 so grid is non-empty
    max_tokens_padded = max(max_tokens_per_expert, 16)

    grid1 = lambda meta: (
        triton.cdiv(max_tokens_padded, meta['BLOCK_M']),
        triton.cdiv(N1, meta['BLOCK_N']),
        E_LOCAL,
    )
    _grouped_gemm1_swiglu[grid1](
        sorted_A,       sorted_A_scale,
        gemm1_weights,  gemm1_weights_scale,
        gemm1_out,
        expert_offsets,
        N1, H,
        max_tokens_padded,
        sorted_A.stride(0),       sorted_A.stride(1),
        sorted_A_scale.stride(0), sorted_A_scale.stride(1),
        gemm1_weights.stride(0),  gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        gemm1_out.stride(0),      gemm1_out.stride(1),
    )

    # ── 5. SwiGLU in-place: gemm1_out[:,I:2I] → silu(up)*gate, stored in [:, :I] ──
    BLOCK_I = 128
    _swiglu_inplace[(triton.cdiv(total_tokens, 16), triton.cdiv(I, BLOCK_I))](
        gemm1_out,
        total_tokens,
        I, 16, BLOCK_I,
    )
    # Now gemm1_out[:, :I] holds the SwiGLU result; rest is garbage.
    # Pass gemm1_out directly to GEMM2 with stride (2*I, 1) so we read only
    # the first I cols without an extra copy.

    # ── 6. Grouped GEMM2 → [total_tokens, H] f32 ─────────────────────────────
    gemm2_out = torch.empty((total_tokens, H), dtype=torch.float32, device=device)

    grid2 = lambda meta: (
        triton.cdiv(max_tokens_padded, meta['BLOCK_M']),
        triton.cdiv(H, meta['BLOCK_N']),
        E_LOCAL,
    )
    _grouped_gemm2[grid2](
        gemm1_out,               # base ptr; stride_am=2*I reads only first I cols
        gemm2_weights,  gemm2_weights_scale,
        gemm2_out,
        expert_offsets,
        H, I,
        max_tokens_padded,
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
