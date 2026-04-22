"""
Sparse Attention Triton Kernel V1.

Grid: (num_tokens, num_qo_heads) — one program per (token, head) pair.
Uses online softmax (flash-attention style) for numerical stability.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_kernel(
    # Pointers
    q_nope_ptr,      # [num_tokens, num_heads, 512] bf16
    q_pe_ptr,        # [num_tokens, num_heads, 64]  bf16
    ckv_ptr,         # [total_kv_tokens, 512]       bf16 (flattened)
    kpe_ptr,         # [total_kv_tokens, 64]        bf16 (flattened)
    indices_ptr,     # [num_tokens, 2048]            int32
    output_ptr,      # [num_tokens, num_heads, 512] bf16
    lse_ptr,         # [num_tokens, num_heads]       f32
    # Scalar
    sm_scale,
    # Strides for q_nope: [num_tokens, num_heads, head_dim_ckv]
    stride_qn_t, stride_qn_h, stride_qn_d,
    # Strides for q_pe: [num_tokens, num_heads, head_dim_kpe]
    stride_qp_t, stride_qp_h, stride_qp_d,
    # Strides for ckv (flattened): [total_tokens, head_dim_ckv]
    stride_ckv_tok, stride_ckv_d,
    # Strides for kpe (flattened): [total_tokens, head_dim_kpe]
    stride_kpe_tok, stride_kpe_d,
    # Strides for indices: [num_tokens, topk]
    stride_idx_t, stride_idx_k,
    # Strides for output: [num_tokens, num_heads, head_dim_ckv]
    stride_o_t, stride_o_h, stride_o_d,
    # Strides for lse: [num_tokens, num_heads]
    stride_lse_t, stride_lse_h,
    # Constants
    HEAD_DIM_CKV: tl.constexpr,   # 512
    HEAD_DIM_KPE: tl.constexpr,   # 64
    TOPK: tl.constexpr,           # 2048
    BLOCK_N: tl.constexpr,        # num KV tokens per iteration
):
    # Which (token, head) this program handles
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    # ---- Step 1: Load query vectors ----
    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)  # [0..511]
    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE)  # [0..63]

    # q_nope[t, h, :] -> [512] float32
    q_nope = tl.load(
        q_nope_ptr + pid_t * stride_qn_t + pid_h * stride_qn_h + offs_d_ckv * stride_qn_d
    ).to(tl.float32)

    # q_pe[t, h, :] -> [64] float32
    q_pe = tl.load(
        q_pe_ptr + pid_t * stride_qp_t + pid_h * stride_qp_h + offs_d_kpe * stride_qp_d
    ).to(tl.float32)

    # ---- Step 2: Init online softmax state ----
    m = tl.full([], float("-inf"), dtype=tl.float32)  # running max (scalar)
    l = tl.zeros([], dtype=tl.float32)                # running sum (scalar)
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)  # weighted accumulator [512]

    # ---- Step 3: Loop over sparse indices in blocks ----
    for start_n in tl.static_range(0, TOPK, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # Load indices for this block
        idx = tl.load(
            indices_ptr + pid_t * stride_idx_t + offs_n * stride_idx_k,
            mask=offs_n < TOPK,
            other=-1,
        )
        valid = idx != -1  # [BLOCK_N] mask

        # Gather Kc[idx, :] -> [BLOCK_N, 512]
        # Address: ckv_ptr + idx * stride_ckv_tok + d * stride_ckv_d
        kc = tl.load(
            ckv_ptr + idx[:, None] * stride_ckv_tok + offs_d_ckv[None, :] * stride_ckv_d,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        # Gather Kp[idx, :] -> [BLOCK_N, 64]
        kp = tl.load(
            kpe_ptr + idx[:, None] * stride_kpe_tok + offs_d_kpe[None, :] * stride_kpe_d,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        # Compute logits: dot(q_nope, Kc[i]) + dot(q_pe, Kp[i]) for each i
        # q_nope is [512], kc is [BLOCK_N, 512] -> element-wise mul + sum over dim 1
        logits_ckv = tl.sum(q_nope[None, :] * kc, axis=1)  # [BLOCK_N]
        logits_kpe = tl.sum(q_pe[None, :] * kp, axis=1)    # [BLOCK_N]
        logits = (logits_ckv + logits_kpe) * sm_scale       # [BLOCK_N]

        # Mask out invalid positions
        logits = tl.where(valid, logits, float("-inf"))

        # ---- Online softmax update ----
        # New block max
        block_max = tl.max(logits, axis=0)        # scalar
        m_new = tl.maximum(m, block_max)

        # Rescale old state
        alpha = tl.exp(m - m_new)                  # scalar
        l = l * alpha
        acc = acc * alpha                          # [512]

        # New contributions
        p = tl.exp(logits - m_new)                 # [BLOCK_N]
        p = tl.where(valid, p, 0.0)               # zero out invalid
        l = l + tl.sum(p, axis=0)                  # scalar

        # Accumulate: acc += sum_i( p[i] * Kc[i, :] )
        # p is [BLOCK_N], kc is [BLOCK_N, 512] -> broadcast p to [BLOCK_N, 512]
        acc += tl.sum(p[:, None] * kc, axis=0)     # [512]

        m = m_new

    # ---- Step 4: Finalize and write outputs ----
    # Normalize accumulator
    acc = acc / l

    # Store output[t, h, :] as bf16
    tl.store(
        output_ptr + pid_t * stride_o_t + pid_h * stride_o_h + offs_d_ckv * stride_o_d,
        acc.to(tl.bfloat16),
    )

    # Compute and store LSE (base-2 log)
    # lse = (m + ln(l)) / ln(2)
    LOG2E: tl.constexpr = 1.4426950408889634
    lse_val = (m + tl.log(l)) * LOG2E
    tl.store(
        lse_ptr + pid_t * stride_lse_t + pid_h * stride_lse_h,
        lse_val,
    )


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    """Python wrapper — flattens caches and launches the Triton kernel."""
    num_tokens, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    # Flatten paged KV caches: [num_pages, page_size, dim] -> [total_tokens, dim]
    ckv_flat = ckv_cache.reshape(-1, head_dim_ckv)
    kpe_flat = kpe_cache.reshape(-1, head_dim_kpe)

    BLOCK_N = 64

    grid = (num_tokens, num_heads)

    _sparse_attn_kernel[grid](
        q_nope, q_pe,
        ckv_flat, kpe_flat,
        sparse_indices,
        output, lse,
        sm_scale,
        # q_nope strides
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        # q_pe strides
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        # ckv strides
        ckv_flat.stride(0), ckv_flat.stride(1),
        # kpe strides
        kpe_flat.stride(0), kpe_flat.stride(1),
        # indices strides
        sparse_indices.stride(0), sparse_indices.stride(1),
        # output strides
        output.stride(0), output.stride(1), output.stride(2),
        # lse strides
        lse.stride(0), lse.stride(1),
        # constants
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        TOPK=topk,
        BLOCK_N=BLOCK_N,
    )
