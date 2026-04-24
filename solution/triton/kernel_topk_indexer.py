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

import os
import torch
import triton
import triton.language as tl

# Set DEBUG_TOPK=1 in the Modal env to make `run` also compute the reference
# (PyTorch ground truth) path and print a diff of the first few mismatched
# positions per batch. Stdout is captured by the framework and surfaced in
# run_modal.py's "First failure log" block.
_DEBUG = os.environ.get("DEBUG_TOPK", "0") == "1"


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

    # int32 page id * stride (can be ~8192) overflows int32 on large caches.
    # Promote to int64 before the address math.
    global_page_id = tl.load(
        BlockTable_ptr + batch_id * stride_bt_b + page_block_id * stride_bt_p
    ).to(tl.int64)

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

        # ReLU, but preserve NaN to match PyTorch torch.relu semantics.
        # tl.maximum(nan, 0.0) returns 0 on some targets (IEEE maxnum rule),
        # which causes us to diverge from the reference when K contains NaN.
        is_nan = dot != dot
        dot = tl.where(is_nan, dot, tl.maximum(dot, 0.0))

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
    # View the FP8 bytes as uint8 for easier manipulation (and to avoid sign-extension issues).
    k_uint8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k_uint8.shape
    # Erase scale Factor
    head_dim = head_dim_sf - 4  # 128

    kv_flat = k_uint8.view(num_pages, page_size * head_dim_sf)

    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [np, ps, 1]

    return fp8_float * scale


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """DPS entry point — Level 2a: Triton-accelerated score computation.

    FP8 dequant still done in PyTorch (same as Level 1), but score computation
    (Q @ K.T + ReLU + weighted sum over heads) fused into a single Triton kernel
    over ALL batches + pages in parallel. Top-k still per-batch (preserves
    numerical match with the reference).
    """
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    _, page_size, _, _ = k_index_cache_fp8.shape
    topk = topk_indices.shape[1]
    max_num_pages = block_table.shape[1]
    max_seq_len = max_num_pages * page_size
    device = q_index_fp8.device

    q = q_index_fp8.to(torch.float32).contiguous()                     # [B, H, D]
    K_all = _dequant_fp8_kv_cache(k_index_cache_fp8).contiguous()      # [num_pages, ps, D]
    w = weights.contiguous()
    bt = block_table.contiguous()

    # Allocate scores buffer. Pre-fill with -inf so any unwritten slot
    # (early-return pages, out-of-range tail) can never win top-k.
    scores = torch.full(
        (batch_size, max_seq_len), float("-inf"),
        dtype=torch.float32, device=device,
    )

    # Launch fused score kernel: one program per (page, batch).
    grid = (max_num_pages, batch_size)
    _fused_score_kernel[grid](
        q, K_all, w, bt, seq_lens, scores,
        q.stride(0), q.stride(1), q.stride(2),
        K_all.stride(0), K_all.stride(1), K_all.stride(2),
        w.stride(0), w.stride(1),
        bt.stride(0), bt.stride(1),
        scores.stride(0), scores.stride(1),
        num_heads=num_index_heads,
        head_dim=index_head_dim,
        page_size=page_size,
        BLOCK_D=64,
    )

    # Top-k + index remap: still per-batch, identical to the reference.
    seq_lens_cpu = seq_lens.tolist()
    topk_indices.fill_(-1)

    for b in range(batch_size):
        seq_len = seq_lens_cpu[b]
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        final_scores = scores[b, :seq_len]                            # [seq_len]

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    if _DEBUG:
        _debug_diff_against_reference(
            q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table,
            topk_indices, scores,
        )


def _reference_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices_out):
    """Verbatim PyTorch reference (from definition JSON). Writes into
    topk_indices_out. Used only by the debug path — must match the framework's
    correctness ground truth bit-for-bit (Level 0 / golden version)."""
    batch_size, _, index_head_dim = q_index_fp8.shape
    _, page_size, _, _ = k_index_cache_fp8.shape
    topk = topk_indices_out.shape[1]

    q = q_index_fp8.to(torch.float32)
    K_all = _dequant_fp8_kv_cache(k_index_cache_fp8)

    topk_indices_out.fill_(-1)
    seq_lens_cpu = seq_lens.tolist()

    ref_scores_per_batch = []  # keep for diff printing

    for b in range(batch_size):
        seq_len = seq_lens_cpu[b]
        if seq_len == 0:
            ref_scores_per_batch.append(None)
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        K_paged = K_all[page_indices]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]

        q_b = q[b]
        raw = q_b @ K.T
        relu = torch.relu(raw)
        w = weights[b]
        final_scores = (relu * w[:, None]).sum(dim=0)
        ref_scores_per_batch.append(final_scores.detach().clone())

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices_out[b, :actual_topk] = topk_tokens.to(torch.int32)

    return ref_scores_per_batch


def _debug_diff_against_reference(
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table,
    our_topk_indices, our_scores,
):
    """Run the verbatim reference and print a per-batch diff with our output.

    Stdout is captured by flashinfer-bench into the `log` field of the trace,
    which run_modal.py dumps on first failure.
    """
    batch_size = q_index_fp8.shape[0]
    ref_topk = torch.full_like(our_topk_indices, -1)
    ref_scores_per_batch = _reference_run(
        q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, ref_topk
    )

    our_ti = our_topk_indices.detach().cpu()
    ref_ti = ref_topk.detach().cpu()
    seq_lens_cpu = seq_lens.tolist()

    print("\n=== DEBUG_TOPK DIFF (Triton vs PyTorch reference) ===")
    print(f"batch_size={batch_size}, shape={tuple(our_ti.shape)}")

    any_diff = False
    for b in range(batch_size):
        seq_len = seq_lens_cpu[b]
        if seq_len == 0:
            continue

        mismatch_mask = our_ti[b] != ref_ti[b]
        n_mismatch = int(mismatch_mask.sum().item())
        if n_mismatch == 0:
            continue
        any_diff = True

        print(f"\n[batch {b}] seq_len={seq_len}, mismatched slots={n_mismatch}")
        # Show first ~8 mismatches.
        mismatch_idx = mismatch_mask.nonzero(as_tuple=True)[0][:8].tolist()
        print(f"  first mismatch slots: {mismatch_idx}")
        for slot in mismatch_idx:
            ours = int(our_ti[b, slot].item())
            theirs = int(ref_ti[b, slot].item())
            print(f"    slot {slot:>4}: ours={ours}  ref={theirs}  (diff={ours - theirs})")

        # Compare raw scores, EXCLUDING positions where either side is NaN.
        # This tells us whether the Triton score kernel matches ref on the
        # finite-value set (which is the part we can actually control).
        ref_scores = ref_scores_per_batch[b]
        if ref_scores is not None:
            ours_scores = our_scores[b, :seq_len].detach().cpu()
            ref_scores_cpu = ref_scores.cpu()

            nan_mask = torch.isnan(ours_scores) | torch.isnan(ref_scores_cpu)
            n_ours_nan = int(torch.isnan(ours_scores).sum().item())
            n_ref_nan = int(torch.isnan(ref_scores_cpu).sum().item())
            n_nan_agree = int((torch.isnan(ours_scores) == torch.isnan(ref_scores_cpu)).all().item())
            print(f"  NaN count: ours={n_ours_nan}  ref={n_ref_nan}  "
                  f"positions_match={bool(n_nan_agree)}")

            # Finite-only score diff.
            finite_diff = (ours_scores - ref_scores_cpu).abs()
            finite_diff = torch.where(nan_mask, torch.zeros_like(finite_diff), finite_diff)
            n_finite = int((~nan_mask).sum().item())
            if n_finite > 0:
                max_err = float(finite_diff.max().item())
                mean_err = float(finite_diff.sum().item() / max(n_finite, 1))
                print(f"  finite-only score error: max_abs={max_err:.3e}  mean_abs={mean_err:.3e}")
                # Top 5 worst finite-only errors.
                worst = torch.topk(finite_diff, k=min(5, finite_diff.numel())).indices.tolist()
                for pos in worst:
                    if nan_mask[pos]:
                        continue
                    o = float(ours_scores[pos].item())
                    r = float(ref_scores_cpu[pos].item())
                    print(f"    token {pos:>5}: our_score={o:.6f}  ref_score={r:.6f}  diff={o - r:.6e}")

    if not any_diff:
        print("All batches match the reference.")
    print("=== END DEBUG_TOPK DIFF ===\n")
