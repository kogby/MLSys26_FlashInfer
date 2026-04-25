"""
TopK Indexer Kernel for FlashInfer Competition — submission build.

Strategy:
- No upfront dequant. Triton kernel reads FP8 bytes + per-token f32 scales
  directly from the paged KV cache, performs an FP8 Tensor Core matmul via
  tl.dot(fp8_Q, fp8_K.T), then applies per-token scale ONCE at the end.
- Math identity used to keep FP8 matmul clean:
    score[t] = sum_h weights[h] * ReLU( Q[h] . K[t] * scale[t] )
             = sum_h weights[h] * ReLU( scale[t] * (Q[h] . K[t]) )
             = sum_h weights[h] * max(0, scale[t] * acc_fp8[h, t])
  Scale must be applied INSIDE the ReLU (scale can be negative because it
  comes from random bytes in the test data).
- Saves 4x HBM traffic on K (no float32 intermediate) and uses sm_100
  Tensor Cores for the 64x128x64 MMA.

Definition: dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_score_kernel(
    Q_fp8_ptr,
    KV_bytes_ptr,
    W_ptr,
    BlockTable_ptr,
    SeqLens_ptr,
    Scores_ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_w_b, stride_w_h,
    stride_bt_b, stride_bt_p,
    stride_scores_b, stride_scores_t,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    head_dim_sf: tl.constexpr,
):
    page_block_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    seq_len = tl.load(SeqLens_ptr + batch_id)
    num_pages_for_seq = (seq_len + page_size - 1) // page_size
    if page_block_id >= num_pages_for_seq:
        return

    # int64 page id to avoid overflow in byte-address math.
    global_page_id = tl.load(
        BlockTable_ptr + batch_id * stride_bt_b + page_block_id * stride_bt_p
    ).to(tl.int64)

    token_start = page_block_id * page_size
    valid_tokens = tl.minimum(seq_len - token_start, page_size)

    tok_offs = tl.arange(0, page_size)
    tok_mask = tok_offs < valid_tokens
    d_offs = tl.arange(0, head_dim)
    h_offs = tl.arange(0, num_heads)

    q_addrs = batch_id * stride_q_b + h_offs[:, None] * stride_q_h + d_offs[None, :] * stride_q_d
    q_fp8 = tl.load(Q_fp8_ptr + q_addrs)

    page_base = global_page_id * (page_size * head_dim_sf)
    k_byte_addrs = page_base + tok_offs[:, None] * head_dim + d_offs[None, :]
    k_bytes = tl.load(
        KV_bytes_ptr + k_byte_addrs,
        mask=tok_mask[:, None],
        other=0,
    )
    k_fp8 = k_bytes.to(tl.float8e4nv, bitcast=True)

    scale_f32_base = (page_base + page_size * head_dim) // 4
    scale_ptr = KV_bytes_ptr.to(tl.pointer_type(tl.float32), bitcast=True)
    scale = tl.load(scale_ptr + scale_f32_base + tok_offs, mask=tok_mask, other=0.0)

    acc = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)

    # Scale-then-ReLU; preserve NaN to match reference semantics.
    scaled = acc * scale[None, :]
    is_nan = scaled != scaled
    relu = tl.where(is_nan, scaled, tl.maximum(scaled, 0.0))

    w = tl.load(W_ptr + batch_id * stride_w_b + h_offs * stride_w_h)
    final = tl.sum(relu * w[:, None], axis=0)

    out_scores = tl.where(tok_mask, final, float("-inf"))
    out_offs = batch_id * stride_scores_b + (token_start + tok_offs) * stride_scores_t
    tl.store(Scores_ptr + out_offs, out_scores)


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """DPS entry point: FP8 Tensor Core score kernel + batched topk."""
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    _, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    topk = topk_indices.shape[1]
    max_num_pages = block_table.shape[1]
    max_seq_len = max_num_pages * page_size
    device = q_index_fp8.device

    q = q_index_fp8.contiguous()
    kv_bytes = k_index_cache_fp8.view(torch.uint8).contiguous().view(-1)
    w = weights.contiguous()
    bt = block_table.contiguous()

    scores = torch.full(
        (batch_size, max_seq_len), float("-inf"),
        dtype=torch.float32, device=device,
    )

    grid = (max_num_pages, batch_size)
    _fused_score_kernel[grid](
        q, kv_bytes, w, bt, seq_lens, scores,
        q.stride(0), q.stride(1), q.stride(2),
        w.stride(0), w.stride(1),
        bt.stride(0), bt.stride(1),
        scores.stride(0), scores.stride(1),
        num_heads=num_index_heads,
        head_dim=index_head_dim,
        page_size=page_size,
        head_dim_sf=head_dim_sf,
    )

    # Batched topk + index remap — fully on GPU, no sync, no Python loop.
    # Padding positions are -inf in `scores`, so a single batched topk over
    # [B, max_seq_len] gives correct ordering for every batch even when
    # seq_len < topk; trailing slots beyond seq_len are masked to -1 below.
    topk_indices.fill_(-1)
    k_eff = min(topk, max_seq_len)
    _, topk_idx = torch.topk(scores, k=k_eff, dim=1)

    page_idx_per_token = topk_idx // page_size
    offset_per_token = topk_idx % page_size
    global_page_idx = block_table.to(torch.long).gather(1, page_idx_per_token)
    topk_tokens = (global_page_idx * page_size + offset_per_token).to(torch.int32)

    valid_mask = torch.arange(k_eff, device=device) < seq_lens.unsqueeze(1)
    topk_indices[:, :k_eff] = torch.where(valid_mask, topk_tokens, torch.full_like(topk_tokens, -1))
