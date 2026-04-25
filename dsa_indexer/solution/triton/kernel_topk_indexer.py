"""
TopK Indexer Kernel for FlashInfer Competition — Level 3.

Strategy:
- No upfront dequant. Triton kernel reads FP8 bytes + per-token f32 scales
  directly from the paged KV cache, performs an FP8 Tensor Core matmul via
  tl.dot(fp8_Q, fp8_K.T), then applies per-token scale ONCE at the end.
- Math identity used to keep FP8 matmul clean:
    score[t] = sum_h weights[h] * ReLU( Q[h] . K[t] * scale[t] )
             = sum_h weights[h] * ReLU( scale[t] * (Q[h] . K[t]) )
             = sum_h weights[h] * max(0, scale[t] * acc_fp8[h, t])
  This matches idxerv3's observation that scale must be applied INSIDE the
  ReLU (scale can be negative because it comes from random bytes in the
  test data).
- Saves 4x HBM traffic on K (no float32 intermediate) and uses sm_100
  Tensor Cores for the 64x128x64 MMA.

Definition: dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
"""

import os
import torch
import triton
import triton.language as tl

# Set FIB_DEBUG=1 in the Modal env (via `--debug True` on run_modal.py) to make
# `run` also compute the reference (PyTorch ground truth) path and print a diff
# of the first few mismatched positions per batch. Stdout is captured by the
# framework and surfaced in run_modal.py's "First failure log" block.
_DEBUG = os.environ.get("FIB_DEBUG", "0") == "1"

# Set FIB_PROFILE=1 in the Modal env (via `--profile True` on run_modal.py) to
# print per-stage CUDA-event timings for every `run()` call. Uses
# torch.cuda.Event (no CUPTI, no profiler).
_PROFILE = os.environ.get("FIB_PROFILE", "0") == "1"

# Benchmark calls run() ~515x per workload. Limit prints to a small middle
# slice so stdout stays readable.
_profile_ctx = {"n": 0, "min": 3, "max": 8}


def _mk_events(n: int):
    return [torch.cuda.Event(enable_timing=True) for _ in range(n)]


@triton.jit
def _fused_score_kernel(
    # Inputs
    Q_fp8_ptr,        # [batch_size, num_heads, head_dim] float8_e4m3fn
    KV_bytes_ptr,     # flattened byte pool: [num_pages * page_size * head_dim_sf] uint8
                      # Per page (8448 bytes): first page_size*head_dim (8192) fp8 data bytes,
                      # then page_size*4 (256) float32 scale bytes.
    W_ptr,            # [batch_size, num_heads] float32
    BlockTable_ptr,   # [batch_size, max_num_pages] int32
    SeqLens_ptr,      # [batch_size] int32
    # Output
    Scores_ptr,       # [batch_size, max_seq_len] float32
    # Strides
    stride_q_b, stride_q_h, stride_q_d,
    stride_w_b, stride_w_h,
    stride_bt_b, stride_bt_p,
    stride_scores_b, stride_scores_t,
    # Constants
    num_heads: tl.constexpr,      # 64
    head_dim: tl.constexpr,       # 128
    page_size: tl.constexpr,      # 64
    head_dim_sf: tl.constexpr,    # 132  (128 fp8 + 4 scale bytes per token row)
):
    """FP8 Tensor Core score kernel.

    Each program computes scores for one page of one batch element via:
        acc[h, t] = sum_d Q_fp8[h, d] * K_fp8[t, d]    (Tensor Core FP8 MMA)
        scored[h, t] = max(0, acc[h, t] * scale[t])    (ReLU after scale)
        score[t] = sum_h weights[h] * scored[h, t]
    """
    page_block_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    seq_len = tl.load(SeqLens_ptr + batch_id)
    num_pages_for_seq = (seq_len + page_size - 1) // page_size
    if page_block_id >= num_pages_for_seq:
        return

    # Promote page id to int64 to avoid overflow in large byte-address math
    # (global_page_id * 8448 can exceed int32 easily).
    global_page_id = tl.load(
        BlockTable_ptr + batch_id * stride_bt_b + page_block_id * stride_bt_p
    ).to(tl.int64)

    token_start = page_block_id * page_size
    valid_tokens = tl.minimum(seq_len - token_start, page_size)

    tok_offs = tl.arange(0, page_size)      # [64]
    tok_mask = tok_offs < valid_tokens      # [64] bool
    d_offs = tl.arange(0, head_dim)         # [128]
    h_offs = tl.arange(0, num_heads)        # [64]

    # ── Load Q tile: [num_heads, head_dim] FP8 ──
    # Q dtype is float8_e4m3fn already (no bitcast needed).
    q_addrs = batch_id * stride_q_b + h_offs[:, None] * stride_q_h + d_offs[None, :] * stride_q_d
    q_fp8 = tl.load(Q_fp8_ptr + q_addrs)    # [64, 128] fp8

    # ── Load K tile (FP8 bytes): [page_size, head_dim] uint8 → bitcast to fp8 ──
    page_base = global_page_id * (page_size * head_dim_sf)
    # FP8 data region is the first page_size*head_dim bytes of the page.
    k_byte_addrs = page_base + tok_offs[:, None] * head_dim + d_offs[None, :]
    k_bytes = tl.load(
        KV_bytes_ptr + k_byte_addrs,
        mask=tok_mask[:, None],
        other=0,  # padding tokens read 0 (won't contaminate MMA; masked out later)
    )
    k_fp8 = k_bytes.to(tl.float8e4nv, bitcast=True)  # [64, 128] fp8

    # ── Load per-token scale: [page_size] float32 ──
    # Scale region starts at page_base + page_size*head_dim bytes and holds
    # page_size float32 values (one per token). Use a reinterpreted f32 ptr.
    scale_f32_base = (page_base + page_size * head_dim) // 4
    scale_ptr = KV_bytes_ptr.to(tl.pointer_type(tl.float32), bitcast=True)
    scale = tl.load(scale_ptr + scale_f32_base + tok_offs, mask=tok_mask, other=0.0)

    # ── FP8 Tensor Core matmul: Q @ K.T = [num_heads, page_size] ──
    acc = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)  # [64, 64] f32

    # ── Apply scale per token (columns), then ReLU (per element) ──
    # scale can be negative / NaN / inf — same convention as reference. ReLU
    # clips negatives; NaN survives to match ref (tl.maximum alone may drop NaN).
    scaled = acc * scale[None, :]                       # [64, 64]
    is_nan = scaled != scaled
    relu   = tl.where(is_nan, scaled, tl.maximum(scaled, 0.0))

    # ── Weighted sum over heads ──
    w = tl.load(W_ptr + batch_id * stride_w_b + h_offs * stride_w_h)  # [64]
    final = tl.sum(relu * w[:, None], axis=0)           # [64]

    # ── Store: padding positions as -inf so they never win top-k ──
    out_scores = tl.where(tok_mask, final, float("-inf"))
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
    """DPS entry point — Level 3: FP8 Tensor Core kernel, no upfront dequant.

    Kernel reads FP8 bytes + per-token scales directly from the paged cache
    and runs an FP8 MMA via tl.dot. Scale is applied inside ReLU (after MMA)
    to preserve the reference semantics (scales can be negative / NaN).
    """
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    _, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    topk = topk_indices.shape[1]
    max_num_pages = block_table.shape[1]
    max_seq_len = max_num_pages * page_size
    device = q_index_fp8.device

    # Decide whether to profile THIS call. Only print the middle iterations
    # (after JIT warmup, before end) to avoid washing out stdout.
    profile_this_call = False
    if _PROFILE:
        n = _profile_ctx["n"]
        _profile_ctx["n"] = n + 1
        if _profile_ctx["min"] <= n < _profile_ctx["max"]:
            profile_this_call = True

    if profile_this_call:
        e_start, e_prep, e_alloc, e_kernel, e_sync, e_topk = _mk_events(6)
        e_start.record()

    # Stage 1: input contiguous / view prep.
    q = q_index_fp8.contiguous()
    kv_bytes = k_index_cache_fp8.view(torch.uint8).contiguous().view(-1)
    w = weights.contiguous()
    bt = block_table.contiguous()
    if profile_this_call:
        e_prep.record()

    # Stage 2: allocate scores buffer pre-filled with -inf.
    scores = torch.full(
        (batch_size, max_seq_len), float("-inf"),
        dtype=torch.float32, device=device,
    )
    if profile_this_call:
        e_alloc.record()

    # Stage 3: Triton FP8 Tensor Core score kernel.
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
    if profile_this_call:
        e_kernel.record()

    # Stage 4: batched topk + index remap — fully on GPU, no sync, no Python loop.
    # Padding positions in `scores` are -inf (set inside the kernel), so a single
    # batched topk over the full [B, max_seq_len] array gives correct ordering for
    # every batch even when seq_len < topk; trailing slots beyond seq_len are
    # masked to -1 below.
    # Some workloads have max_seq_len < topk; clamp k so torch.topk doesn't OOR.
    if profile_this_call:
        e_sync.record()

    topk_indices.fill_(-1)
    k_eff = min(topk, max_seq_len)
    _, topk_idx = torch.topk(scores, k=k_eff, dim=1)                 # [B, k_eff]

    page_idx_per_token = topk_idx // page_size                       # [B, k_eff]
    offset_per_token = topk_idx % page_size                          # [B, k_eff]
    global_page_idx = block_table.to(torch.long).gather(1, page_idx_per_token)
    topk_tokens = (global_page_idx * page_size + offset_per_token).to(torch.int32)

    valid_mask = torch.arange(k_eff, device=device) < seq_lens.unsqueeze(1)
    topk_indices[:, :k_eff] = torch.where(valid_mask, topk_tokens, torch.full_like(topk_tokens, -1))

    if profile_this_call:
        e_topk.record()
        torch.cuda.synchronize()
        total = e_start.elapsed_time(e_topk)
        t_prep = e_start.elapsed_time(e_prep)
        t_alloc = e_prep.elapsed_time(e_alloc)
        t_kernel = e_alloc.elapsed_time(e_kernel)
        t_sync = e_kernel.elapsed_time(e_sync)
        t_topk = e_sync.elapsed_time(e_topk)
        print(
            f"[PROFILE #{_profile_ctx['n']-1}] "
            f"B={batch_size} max_pages={max_num_pages} "
            f"seq_lens={seq_lens.tolist()[:4]}{'...' if batch_size > 4 else ''} "
            f"| total={total:.3f}ms  "
            f"prep={t_prep:.3f}  alloc={t_alloc:.3f}  "
            f"kernel={t_kernel:.3f}  sync={t_sync:.3f}  topk={t_topk:.3f}  "
            f"(ms)",
            flush=True,
        )

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

    print("\n=== FIB_DEBUG DIFF (Triton vs PyTorch reference) ===")
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
    print("=== END FIB_DEBUG DIFF ===\n")
