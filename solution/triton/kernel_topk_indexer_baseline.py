"""
TopK Indexer Baseline for FlashInfer Competition.

DPS-style baseline using PyTorch operations.
Definition: dsa_topk_indexer_fp8_h64_d128_topk2048_ps64

Inputs:
    q_index_fp8:        [batch_size, 64, 128]           fp8   - queries
    k_index_cache_fp8:  [num_pages, 64, 1, 132]         int8  - FP8 KV cache (deep_gemm format)
    weights:            [batch_size, 64]                 f32   - learned per-head weights
    seq_lens:           [batch_size]                     int32 - sequence lengths
    block_table:        [batch_size, max_num_pages]      int32 - page table

Outputs (pre-allocated, written in-place):
    topk_indices:       [batch_size, 2048]               int32 - selected token indices
"""

import torch


def dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV cache from deep_gemm format.

    Input: [num_pages, page_size, 1, 132] int8
    Output: [num_pages, page_size, 128] float32
    """
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4  # 128

    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)

    # FP8 data: first page_size * head_dim bytes
    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    # Scale: last page_size * 4 bytes -> page_size float32 values
    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float * scale


def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """TopK indexer kernel (DPS style)."""
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    # Dequantize inputs
    q = q_index_fp8.to(torch.float32)
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)

    # Initialize output
    topk_indices.fill_(-1)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())

        if seq_len == 0:
            continue

        # Get pages for this sequence
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        # Gather K from pages
        K_paged = K_all[page_indices]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]

        # Compute attention scores with ReLU
        q_b = q[b]
        scores = q_b @ K.T
        scores_relu = torch.relu(scores)

        # Apply learned weights and sum across heads
        w = weights[b]
        weighted_scores = scores_relu * w[:, None]
        final_scores = weighted_scores.sum(dim=0)

        # Select top-K
        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        # Convert to global token indices
        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)
