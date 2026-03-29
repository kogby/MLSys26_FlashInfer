"""
Sparse Attention Kernel for FlashInfer Competition.

Baseline implementation using PyTorch operations in DPS style.
"""

import math
import torch
import triton
import triton.language as tl


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    """
    Sparse attention kernel (DPS style).

    Inputs:
        q_nope:         [num_tokens, 16, 512]  bfloat16 - queries without positional encoding
        q_pe:           [num_tokens, 16, 64]   bfloat16 - queries with positional encoding
        ckv_cache:      [num_pages, 64, 512]   bfloat16 - compressed KV cache (paged)
        kpe_cache:      [num_pages, 64, 64]    bfloat16 - KV positional encoding cache (paged)
        sparse_indices: [num_tokens, 2048]     int32    - selected token indices (-1 = invalid)
        sm_scale:       float                           - softmax scale factor

    Outputs (pre-allocated, written in-place):
        output:         [num_tokens, 16, 512]  bfloat16 - attention output
        lse:            [num_tokens, 16]       float32  - log-sum-exp (base 2)
    """
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Check constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size
    
    # Flatten paged KV cache to token-level
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    # Initialize outputs
    output.zero_()
    lse.fill_(float("-inf"))

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [2048]

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1
        valid_indices = indices[valid_mask].to(torch.long)

        if valid_indices.numel() == 0:
            continue

        # Gather KV entries for selected tokens
        Kc = Kc_all[valid_indices]  # [num_valid, 512]
        Kp = Kp_all[valid_indices]  # [num_valid, 64]
        qn = q_nope[t].to(torch.float32)  # [16, 512]
        qp = q_pe[t].to(torch.float32)    # [16, 64]

        # Compute attention logits
        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [16, num_valid]
        logits_scaled = logits * sm_scale

        # Compute LSE (base-2 logarithm)
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        # Compute attention output
        attn = torch.softmax(logits_scaled, dim=-1)  # [16, num_valid]
        out = attn @ Kc  # [16, 512]
        output[t] = out.to(torch.bfloat16)
