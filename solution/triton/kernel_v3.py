"""
Sparse Attention Triton Kernel V3 — KV-Split + Hybrid Dispatch.

Builds on V2 (merged-heads, tensor cores) by adding Flash-Decoding style
parallelism over TOPK. Key changes:

1. TOPK=2048 split into NUM_SPLITS=8 chunks of DIM_SPLIT=256 each.
   Grid: (num_tokens,) → (num_tokens, NUM_SPLITS). 8× more blocks, 8× more
   SMs active for small batches.

2. Two-kernel design (Flash-Decoding / Dao 2023):
   - _compute_kernel:  per-split attention, writes partial_out + (m, l)
   - _reduce_kernel:   merges splits via log-sum-exp, writes final output

3. Hybrid dispatch:
   - T < 3: V2 direct path (no split overhead; one kernel launch).
   - T ≥ 3: KV-split path (two kernels, 8× more parallelism).

4. Workspace (partial_out, partial_lse) cached across calls.

"""

import torch
import triton
import triton.language as tl


# ── Problem constants ─────────────────────────────────────────────────────────

NUM_SPLITS_CONST = 8
DIM_SPLIT_CONST = 2048 // NUM_SPLITS_CONST   # 256
T_SMALL_THRESHOLD = 3   # below this, use single-kernel V2 path

# Softmax uses -1.0e30 (not -inf) as "empty" sentinel. Reason: exp(-inf-(-inf))
# = exp(NaN) = NaN propagates through l, acc and corrupts the output. With a
# large finite negative value, exp(-1e30 - (-1e30)) = exp(0) = 1 stays stable,
# and exp(-1e30 - finite) underflows to 0 (same as -inf arithmetically).


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel A: V2 direct — used for T < 3 (no KV-split)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["TOPK"],
)
@triton.jit
def _v2_kernel(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr,
    indices_ptr, output_ptr, lse_ptr,
    sm_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_tok, stride_ckv_d,
    stride_kpe_tok, stride_kpe_d,
    stride_idx_t, stride_idx_k,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)

    offs_h = tl.arange(0, NUM_HEADS)
    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE)

    q_nope = tl.load(
        q_nope_ptr + pid_t * stride_qn_t
        + offs_h[:, None] * stride_qn_h
        + offs_d_ckv[None, :] * stride_qn_d,
    )
    q_pe = tl.load(
        q_pe_ptr + pid_t * stride_qp_t
        + offs_h[:, None] * stride_qp_h
        + offs_d_kpe[None, :] * stride_qp_d,
    )

    # NaN-safe init: -1e30 instead of -inf. See -1.0e30 comment.
    m = tl.full([NUM_HEADS], -1.0e30, dtype=tl.float32)
    l = tl.zeros([NUM_HEADS], dtype=tl.float32)
    acc = tl.zeros([NUM_HEADS, HEAD_DIM_CKV], dtype=tl.float32)

    for start_n in tl.static_range(0, TOPK, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        idx = tl.load(
            indices_ptr + pid_t * stride_idx_t + offs_n * stride_idx_k,
            mask=offs_n < TOPK, other=-1,
        )
        valid = idx >= 0

        kc = tl.load(
            ckv_ptr + idx[:, None] * stride_ckv_tok + offs_d_ckv[None, :] * stride_ckv_d,
            mask=valid[:, None], other=0.0,
        )
        kp = tl.load(
            kpe_ptr + idx[:, None] * stride_kpe_tok + offs_d_kpe[None, :] * stride_kpe_d,
            mask=valid[:, None], other=0.0,
        )

        logits = tl.dot(q_nope, tl.trans(kc), out_dtype=tl.float32)
        logits += tl.dot(q_pe, tl.trans(kp), out_dtype=tl.float32)
        logits = logits * sm_scale
        # Invalid positions mapped to sentinel (not -inf) to avoid NaN when the
        # whole block is invalid.
        logits = tl.where(valid[None, :], logits, -1.0e30)

        block_max = tl.max(logits, axis=1)
        m_new = tl.maximum(m, block_max)
        alpha = tl.exp(m - m_new)
        l = l * alpha
        acc = acc * alpha[:, None]

        p = tl.exp(logits - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        l = l + tl.sum(p, axis=1)
        acc += tl.dot(p.to(tl.bfloat16), kc, out_dtype=tl.float32)
        m = m_new

    # Guard against divide-by-zero (all-invalid token)
    safe_l = tl.maximum(l, 1e-30)
    acc = acc / safe_l[:, None]
    tl.store(
        output_ptr + pid_t * stride_o_t
        + offs_h[:, None] * stride_o_h
        + offs_d_ckv[None, :] * stride_o_d,
        acc.to(tl.bfloat16),
    )

    LOG2E: tl.constexpr = 1.4426950408889634
    lse_val = tl.where(l > 0, (m + tl.log(l)) * LOG2E, float("-inf"))
    tl.store(
        lse_ptr + pid_t * stride_lse_t + offs_h * stride_lse_h,
        lse_val,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel B: per-split compute — used for T ≥ 3
# Grid: (num_tokens, NUM_SPLITS)
# Each program handles ALL heads for ONE split of DIM_SPLIT=256 indices.
# ═══════════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["DIM_SPLIT"],
)
@triton.jit
def _compute_kernel(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr,
    indices_ptr,
    partial_out_ptr, partial_lse_ptr,
    sm_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_tok, stride_ckv_d,
    stride_kpe_tok, stride_kpe_d,
    stride_idx_t, stride_idx_k,
    # partial_out: [T, SPLITS, HEADS, HEAD_DIM_CKV]
    stride_po_t, stride_po_s, stride_po_h, stride_po_d,
    # partial_lse: [T, SPLITS, HEADS, 2]   (0=m, 1=l)
    stride_pl_t, stride_pl_s, stride_pl_h, stride_pl_ml,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    TOPK: tl.constexpr,
    DIM_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    split_start = pid_s * DIM_SPLIT

    offs_h = tl.arange(0, NUM_HEADS)
    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE)

    q_nope = tl.load(
        q_nope_ptr + pid_t * stride_qn_t
        + offs_h[:, None] * stride_qn_h
        + offs_d_ckv[None, :] * stride_qn_d,
    )
    q_pe = tl.load(
        q_pe_ptr + pid_t * stride_qp_t
        + offs_h[:, None] * stride_qp_h
        + offs_d_kpe[None, :] * stride_qp_d,
    )

    # NaN-safe init: -1e30 instead of -inf so empty-split blocks don't produce
    # NaN via exp(-inf - (-inf)). See -1.0e30 comment.
    m = tl.full([NUM_HEADS], -1.0e30, dtype=tl.float32)
    l = tl.zeros([NUM_HEADS], dtype=tl.float32)
    acc = tl.zeros([NUM_HEADS, HEAD_DIM_CKV], dtype=tl.float32)

    # Loop over DIM_SPLIT=256 indices for this split
    for start_n in tl.static_range(0, DIM_SPLIT, BLOCK_N):
        offs_n = split_start + start_n + tl.arange(0, BLOCK_N)
        idx = tl.load(
            indices_ptr + pid_t * stride_idx_t + offs_n * stride_idx_k,
            mask=offs_n < TOPK, other=-1,
        )
        valid = idx >= 0

        kc = tl.load(
            ckv_ptr + idx[:, None] * stride_ckv_tok + offs_d_ckv[None, :] * stride_ckv_d,
            mask=valid[:, None], other=0.0,
        )
        kp = tl.load(
            kpe_ptr + idx[:, None] * stride_kpe_tok + offs_d_kpe[None, :] * stride_kpe_d,
            mask=valid[:, None], other=0.0,
        )

        logits = tl.dot(q_nope, tl.trans(kc), out_dtype=tl.float32)
        logits += tl.dot(q_pe, tl.trans(kp), out_dtype=tl.float32)
        logits = logits * sm_scale
        logits = tl.where(valid[None, :], logits, -1.0e30)

        block_max = tl.max(logits, axis=1)
        m_new = tl.maximum(m, block_max)
        alpha = tl.exp(m - m_new)
        l = l * alpha
        acc = acc * alpha[:, None]

        p = tl.exp(logits - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        l = l + tl.sum(p, axis=1)
        acc += tl.dot(p.to(tl.bfloat16), kc, out_dtype=tl.float32)
        m = m_new

    # Write UNNORMALIZED acc — reduce kernel normalizes via merged LSE.
    tl.store(
        partial_out_ptr
        + pid_t * stride_po_t + pid_s * stride_po_s
        + offs_h[:, None] * stride_po_h
        + offs_d_ckv[None, :] * stride_po_d,
        acc,
    )

    # Write (m, l) pair per head
    base_lse = partial_lse_ptr + pid_t * stride_pl_t + pid_s * stride_pl_s
    tl.store(base_lse + offs_h * stride_pl_h + 0 * stride_pl_ml, m)
    tl.store(base_lse + offs_h * stride_pl_h + 1 * stride_pl_ml, l)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel C: reduce — merge NUM_SPLITS partials via log-sum-exp
# Grid: (num_tokens,) — one program per token, handles all heads.
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _reduce_kernel(
    partial_out_ptr, partial_lse_ptr,
    output_ptr, lse_ptr,
    stride_po_t, stride_po_s, stride_po_h, stride_po_d,
    stride_pl_t, stride_pl_s, stride_pl_h, stride_pl_ml,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    NUM_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
):
    pid_t = tl.program_id(0)

    offs_h = tl.arange(0, NUM_HEADS)
    offs_d = tl.arange(0, HEAD_DIM_CKV)
    offs_s = tl.arange(0, NUM_SPLITS)

    # Load all (m, l) pairs: [SPLITS, HEADS]
    m_all = tl.load(
        partial_lse_ptr + pid_t * stride_pl_t
        + offs_s[:, None] * stride_pl_s
        + offs_h[None, :] * stride_pl_h
        + 0 * stride_pl_ml,
    )
    l_all = tl.load(
        partial_lse_ptr + pid_t * stride_pl_t
        + offs_s[:, None] * stride_pl_s
        + offs_h[None, :] * stride_pl_h
        + 1 * stride_pl_ml,
    )

    g_max = tl.max(m_all, axis=0)                 # [HEADS]
    scale_all = tl.exp(m_all - g_max[None, :])    # [SPLITS, HEADS]
    g_sum = tl.sum(l_all * scale_all, axis=0)     # [HEADS]

    # Weighted sum of partials — loop over splits to bound register usage.
    acc = tl.zeros([NUM_HEADS, HEAD_DIM_CKV], dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(
            partial_lse_ptr + pid_t * stride_pl_t + s * stride_pl_s
            + offs_h * stride_pl_h + 0 * stride_pl_ml,
        )
        scale_s = tl.exp(m_s - g_max)
        po = tl.load(
            partial_out_ptr + pid_t * stride_po_t + s * stride_po_s
            + offs_h[:, None] * stride_po_h
            + offs_d[None, :] * stride_po_d,
        )
        acc += scale_s[:, None] * po

    safe_sum = tl.maximum(g_sum, 1e-30)
    out = acc / safe_sum[:, None]

    tl.store(
        output_ptr + pid_t * stride_o_t
        + offs_h[:, None] * stride_o_h
        + offs_d[None, :] * stride_o_d,
        out.to(tl.bfloat16),
    )

    LOG2E: tl.constexpr = 1.4426950408889634
    lse_val = tl.where(g_sum > 0,
                       (g_max + tl.log(g_sum)) * LOG2E,
                       float("-inf"))
    tl.store(
        lse_ptr + pid_t * stride_lse_t + offs_h * stride_lse_h,
        lse_val,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Python wrapper — hybrid dispatch + workspace caching
# ═══════════════════════════════════════════════════════════════════════════════

_WORKSPACE = {}


def _get_workspace(num_tokens, num_heads, head_dim_ckv, device):
    key = (num_tokens, num_heads, head_dim_ckv, str(device))
    ws = _WORKSPACE.get(key)
    if ws is None:
        partial_out = torch.empty(
            num_tokens, NUM_SPLITS_CONST, num_heads, head_dim_ckv,
            dtype=torch.float32, device=device,
        )
        partial_lse = torch.empty(
            num_tokens, NUM_SPLITS_CONST, num_heads, 2,
            dtype=torch.float32, device=device,
        )
        ws = (partial_out, partial_lse)
        _WORKSPACE[key] = ws
    return ws


def _run_v2_direct(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                   sm_scale, output, lse):
    num_tokens, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    grid = (num_tokens,)
    _v2_kernel[grid](
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_flat.stride(0), ckv_flat.stride(1),
        kpe_flat.stride(0), kpe_flat.stride(1),
        sparse_indices.stride(0), sparse_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        NUM_HEADS=num_heads,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        TOPK=topk,
    )


def _run_kv_split(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                  sm_scale, output, lse):
    num_tokens, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    partial_out, partial_lse = _get_workspace(
        num_tokens, num_heads, head_dim_ckv, q_nope.device,
    )

    # Compute: (T, NUM_SPLITS) grid
    compute_grid = (num_tokens, NUM_SPLITS_CONST)
    _compute_kernel[compute_grid](
        q_nope, q_pe, ckv_flat, kpe_flat,
        sparse_indices,
        partial_out, partial_lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_flat.stride(0), ckv_flat.stride(1),
        kpe_flat.stride(0), kpe_flat.stride(1),
        sparse_indices.stride(0), sparse_indices.stride(1),
        partial_out.stride(0), partial_out.stride(1),
        partial_out.stride(2), partial_out.stride(3),
        partial_lse.stride(0), partial_lse.stride(1),
        partial_lse.stride(2), partial_lse.stride(3),
        NUM_HEADS=num_heads,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        TOPK=topk,
        DIM_SPLIT=DIM_SPLIT_CONST,
    )

    # Reduce: (T,) grid
    reduce_grid = (num_tokens,)
    _reduce_kernel[reduce_grid](
        partial_out, partial_lse, output, lse,
        partial_out.stride(0), partial_out.stride(1),
        partial_out.stride(2), partial_out.stride(3),
        partial_lse.stride(0), partial_lse.stride(1),
        partial_lse.stride(2), partial_lse.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        NUM_HEADS=num_heads,
        NUM_SPLITS=NUM_SPLITS_CONST,
        HEAD_DIM_CKV=head_dim_ckv,
    )


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
           sm_scale, output, lse):
    """V3 entry — hybrid dispatch.

    Args:
        q_nope:         [T, 16, 512]  bf16
        q_pe:           [T, 16, 64]   bf16
        ckv_cache:      [P, 64, 512]  bf16 (paged)
        kpe_cache:      [P, 64, 64]   bf16 (paged)
        sparse_indices: [T, 2048]     int32 (-1 = invalid)
        sm_scale:       float
        output:         [T, 16, 512]  bf16 (DPS output)
        lse:            [T, 16]       f32  (DPS output)
    """
    num_tokens = q_nope.shape[0]
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])

    if num_tokens < T_SMALL_THRESHOLD:
        _run_v2_direct(q_nope, q_pe, ckv_flat, kpe_flat,
                       sparse_indices, sm_scale, output, lse)
    else:
        _run_kv_split(q_nope, q_pe, ckv_flat, kpe_flat,
                      sparse_indices, sm_scale, output, lse)
