"""
TopK Indexer Kernel for FlashInfer Competition — conservative Step-1 version.

Strategy:
- FP8 dequant is done in PyTorch (same as the reference) — proven correct,
  no Triton FP8 bitcast to debug.
- Triton kernel consumes the dequantised float32 K and fused-computes
  score[b, t] = sum_h( weights[b,h] * ReLU( Q[b,h,:] . K[t,:] ) ) per page.
- Final topk still done via a per-batch PyTorch loop (to be vectorised later).

Definition: dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_score_kernel(
    # Inputs
    Q_ptr,            # [batch_size, num_heads, head_dim] float32
    K_ptr,            # [num_pages, page_size, head_dim] float32 (already dequantised)
    W_ptr,            # [batch_size, num_heads] float32
    BlockTable_ptr,   # [batch_size, max_num_pages] int32
    SeqLens_ptr,      # [batch_size] int32
    # Output
    Scores_ptr,       # [batch_size, max_seq_len] float32
    # Strides (all inputs are asserted contiguous in the host code, so these
    # correspond to row strides of the natural layout).
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_page, stride_k_tok, stride_k_d,
    stride_w_b, stride_w_h,
    stride_bt_b, stride_bt_p,
    stride_scores_b, stride_scores_t,
    # Constants
    num_heads: tl.constexpr,      # 64
    head_dim: tl.constexpr,       # 128
    page_size: tl.constexpr,      # 64
    BLOCK_D: tl.constexpr,        # head_dim block size
):
    """Compute weighted-ReLU scores for one page of one batch element."""
    page_block_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    seq_len = tl.load(SeqLens_ptr + batch_id)
    num_pages_for_seq = (seq_len + page_size - 1) // page_size
    if page_block_id >= num_pages_for_seq:
        return

    global_page_id = tl.load(
        BlockTable_ptr + batch_id * stride_bt_b + page_block_id * stride_bt_p
    )

    token_start = page_block_id * page_size
    valid_tokens = tl.minimum(seq_len - token_start, page_size)

    tok_offs = tl.arange(0, page_size)
    tok_mask = tok_offs < valid_tokens

    acc_scores = tl.zeros([page_size], dtype=tl.float32)

    for h in range(num_heads):
        q_base = batch_id * stride_q_b + h * stride_q_h
        dot = tl.zeros([page_size], dtype=tl.float32)

        for d in tl.static_range(0, head_dim, BLOCK_D):
            d_offs = d + tl.arange(0, BLOCK_D)

            # Q chunk: [BLOCK_D]
            q_chunk = tl.load(Q_ptr + q_base + d_offs * stride_q_d)

            # K chunk: [page_size, BLOCK_D]
            k_addrs = (
                global_page_id * stride_k_page
                + tok_offs[:, None] * stride_k_tok
                + d_offs[None, :] * stride_k_d
            )
            k_chunk = tl.load(K_ptr + k_addrs, mask=tok_mask[:, None], other=0.0)

            dot += tl.sum(k_chunk * q_chunk[None, :], axis=1)

        dot = tl.maximum(dot, 0.0)  # ReLU

        w = tl.load(W_ptr + batch_id * stride_w_b + h * stride_w_h)
        acc_scores += dot * w

    # Write tail tokens as -inf so they lose the top-k race later.
    out_scores = tl.where(tok_mask, acc_scores, float("-inf"))
    out_offs = batch_id * stride_scores_b + (token_start + tok_offs) * stride_scores_t
    tl.store(Scores_ptr + out_offs, out_scores)


def _dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantise deep_gemm FP8 paged cache to float32.

    Input:  [num_pages, page_size, 1, 132] int8  (128 FP8 bytes + 4 scale bytes per row group)
    Output: [num_pages, page_size, 128] float32
    """
    k_uint8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k_uint8.shape
    head_dim = head_dim_sf - 4  # 128

    kv_flat = k_uint8.view(num_pages, page_size * head_dim_sf)

    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [np, ps, 1]

    return fp8_float * scale


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """DPS entry point — Level 1 (minimal): exact copy of the reference loop,
    but replace per-iteration `seq_lens[b].item()` (N GPU->CPU syncs) with
    a single `seq_lens.tolist()` (one sync). Pure Python-overhead win with
    zero numerical change.
    """
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    _, page_size, _, _ = k_index_cache_fp8.shape
    topk = topk_indices.shape[1]

    q = q_index_fp8.to(torch.float32)                  # [B, H, D]
    K_all = _dequant_fp8_kv_cache(k_index_cache_fp8)   # [num_pages, ps, D]

    # One GPU->CPU sync for all seq_lens, instead of N per-iteration syncs.
    seq_lens_cpu = seq_lens.tolist()

    topk_indices.fill_(-1)

    for b in range(batch_size):
        seq_len = seq_lens_cpu[b]
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        K_paged = K_all[page_indices]                                 # [P, ps, D]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]             # [seq_len, D]

        q_b = q[b]                                                    # [H, D]
        scores = q_b @ K.T                                            # [H, seq_len]
        scores_relu = torch.relu(scores)

        w = weights[b]                                                # [H]
        weighted_scores = scores_relu * w[:, None]
        final_scores = weighted_scores.sum(dim=0)                     # [seq_len]

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)
