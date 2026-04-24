"""
Sparse Attention Triton Kernel V2.

Key optimizations over V1:
1. Grid: (num_tokens,) — one program per token handles ALL heads.
   Since KV cache is head-shared (no head dim), V1 loaded each KV block 16x.
   V2 loads it once and broadcasts across heads → 16× HBM reduction.
2. tl.dot for Q@K^T and P@V — uses tensor cores (bf16 inputs, fp32 acc).
3. @triton.autotune over BLOCK_N / num_warps / num_stages.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["TOPK"],
)
@triton.jit
def _sparse_attn_kernel_v2(
    # Pointers
    q_nope_ptr,   # [num_tokens, num_heads, 512]  bf16
    q_pe_ptr,     # [num_tokens, num_heads, 64]   bf16
    ckv_ptr,      # [total_kv_tokens, 512]         bf16 (flattened)
    kpe_ptr,      # [total_kv_tokens, 64]          bf16 (flattened)
    indices_ptr,  # [num_tokens, 2048]              int32
    output_ptr,   # [num_tokens, num_heads, 512]   bf16
    lse_ptr,      # [num_tokens, num_heads]         f32
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
    NUM_HEADS:    tl.constexpr,  # 16
    HEAD_DIM_CKV: tl.constexpr,  # 512
    HEAD_DIM_KPE: tl.constexpr,  # 64
    TOPK:         tl.constexpr,  # 2048
    BLOCK_N:      tl.constexpr,  # tuned: 32/64/128
):
    pid_t = tl.program_id(0)

    offs_h    = tl.arange(0, NUM_HEADS)     # [16]
    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV) # [512]
    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE) # [64]

    # ---- Step 1: Load query vectors for ALL heads ----
    # q_nope[t, :, :] -> [NUM_HEADS, HEAD_DIM_CKV]  bf16
    q_nope = tl.load(
        q_nope_ptr
        + pid_t * stride_qn_t
        + offs_h[:, None]    * stride_qn_h
        + offs_d_ckv[None, :] * stride_qn_d,
    )  # bf16 [16, 512]

    # q_pe[t, :, :] -> [NUM_HEADS, HEAD_DIM_KPE]  bf16
    q_pe = tl.load(
        q_pe_ptr
        + pid_t * stride_qp_t
        + offs_h[:, None]    * stride_qp_h
        + offs_d_kpe[None, :] * stride_qp_d,
    )  # bf16 [16, 64]

    # ---- Step 2: Init online-softmax state (per head) ----
    m   = tl.full([NUM_HEADS], float("-inf"), dtype=tl.float32)  # [16]
    l   = tl.zeros([NUM_HEADS], dtype=tl.float32)                # [16]
    acc = tl.zeros([NUM_HEADS, HEAD_DIM_CKV], dtype=tl.float32)  # [16, 512]

    # ---- Step 3: Loop over sparse KV blocks — load KV ONCE for all heads ----
    for start_n in tl.static_range(0, TOPK, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # Load token indices for this block
        idx = tl.load(
            indices_ptr + pid_t * stride_idx_t + offs_n * stride_idx_k,
            mask=offs_n < TOPK,
            other=-1,
        )  # [BLOCK_N] int32
        valid = idx >= 0  # [BLOCK_N]

        # Gather Kc[idx, :] -> [BLOCK_N, 512]  (ONE load for ALL 16 heads)
        kc = tl.load(
            ckv_ptr
            + idx[:, None]    * stride_ckv_tok
            + offs_d_ckv[None, :] * stride_ckv_d,
            mask=valid[:, None],
            other=0.0,
        )  # bf16 [BLOCK_N, 512]

        # Gather Kp[idx, :] -> [BLOCK_N, 64]
        kp = tl.load(
            kpe_ptr
            + idx[:, None]    * stride_kpe_tok
            + offs_d_kpe[None, :] * stride_kpe_d,
            mask=valid[:, None],
            other=0.0,
        )  # bf16 [BLOCK_N, 64]

        # ---- Compute logits via tl.dot (tensor cores) ----
        # [16, 512] @ [512, BLOCK_N] = [16, BLOCK_N]
        logits = tl.dot(q_nope, tl.trans(kc), out_dtype=tl.float32)
        logits += tl.dot(q_pe,  tl.trans(kp), out_dtype=tl.float32)
        logits = logits * sm_scale  # [16, BLOCK_N]

        # Mask invalid positions
        logits = tl.where(valid[None, :], logits, float("-inf"))

        # ---- Online softmax update (per head) ----
        block_max = tl.max(logits, axis=1)  # [16]
        m_new     = tl.maximum(m, block_max)

        alpha = tl.exp(m - m_new)           # [16]
        l   = l * alpha
        acc = acc * alpha[:, None]          # [16, 512]

        p = tl.exp(logits - m_new[:, None]) # [16, BLOCK_N]
        p = tl.where(valid[None, :], p, 0.0)
        l = l + tl.sum(p, axis=1)           # [16]

        # ---- Value accumulation via tl.dot (tensor cores) ----
        # [16, BLOCK_N] @ [BLOCK_N, 512] = [16, 512]
        acc += tl.dot(p.to(tl.bfloat16), kc, out_dtype=tl.float32)

        m = m_new

    # ---- Step 4: Finalize and write outputs ----
    acc = acc / l[:, None]  # [16, 512]

    # Store output[t, :, :] as bf16
    tl.store(
        output_ptr
        + pid_t * stride_o_t
        + offs_h[:, None]    * stride_o_h
        + offs_d_ckv[None, :] * stride_o_d,
        acc.to(tl.bfloat16),
    )

    # Store LSE (base-2)
    LOG2E: tl.constexpr = 1.4426950408889634
    lse_val = (m + tl.log(l)) * LOG2E  # [16]
    tl.store(
        lse_ptr + pid_t * stride_lse_t + offs_h * stride_lse_h,
        lse_val,
    )


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    """Python wrapper — flattens caches and launches V2 kernel."""
    num_tokens, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    ckv_flat = ckv_cache.reshape(-1, head_dim_ckv)
    kpe_flat = kpe_cache.reshape(-1, head_dim_kpe)

    # Grid: one program per TOKEN (all heads merged)
    grid = (num_tokens,)

    _sparse_attn_kernel_v2[grid](
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
        NUM_HEADS=num_heads,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        TOPK=topk,
    )
