"""
CUTLASS cute port — iter1: basic warp-GEMV attention (single kernel, full TOPK).

GOAL: correctness match vs V4. Grid [T, H, 1], 1024 threads = 32 warps.
Each warp handles ~64 KV tokens. Each lane handles 16 dims of 512-dim ckv.

ALGORITHM (2-pass softmax, simpler for first cute iteration):
  1. Load Q into smem (once per (T, H) block)
  2. Load sparse_indices into smem + count valid
  3. SCORE phase: each warp takes one KV token at a time via round-robin.
     Warp-GEMV: 32 lanes each hold 16 dims, partial sum, warp-shuffle reduce.
     Writes logit to smem_logits[sparse_idx].
  4. Softmax pass 1: block-wide max over valid logits
  5. Softmax pass 2: exp + sum (fused write-back). Divide inline in output loop.
  6. OUTPUT phase: each warp takes one KV token, accumulates weight * V_row
     into per-lane register tensor (16 dims per lane).
  7. Cross-warp reduce: each thread sums 32 warps' contributions for its dim.
"""

import math
import sys
import traceback

import torch

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    CUTLASS_AVAILABLE = True
except ImportError as e:
    CUTLASS_AVAILABLE = False
    _import_err = str(e)
    print(f"[kernel_cute] cutlass import FAILED: {e}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Problem shape constants
# ═══════════════════════════════════════════════════════════════════════════════

NUM_HEADS     = 16
HEAD_DIM_CKV  = 512
HEAD_DIM_KPE  = 64
TOP_K         = 2048
NUM_PAGES     = 8462
PAGE_SIZE     = 64

BLOCK_SIZE    = 1024          # threads per block
NUM_WARPS     = 32            # BLOCK_SIZE // 32
WARP_SIZE     = 32
DIMS_PER_LANE = HEAD_DIM_CKV // WARP_SIZE   # 16  — each lane's slice of ckv

LN2_INV       = 1.4426950408889634   # 1 / ln(2)


if CUTLASS_AVAILABLE:

    @cute.jit
    def _warp_reduce(val: cute.Numeric, op: callable,
                     width: cutlass.Constexpr = 32) -> cute.Numeric:
        # Butterfly reduction via shuffle_sync_bfly.
        for i in range(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val

    @cute.kernel
    def _attn_kernel(
        q_nope:         cute.Tensor,        # (T, H, 512)
        q_pe:           cute.Tensor,        # (T, H, 64)
        ckv_cache:      cute.Tensor,        # (N, 512)  flat
        kpe_cache:      cute.Tensor,        # (N, 64)   flat
        sparse_indices: cute.Tensor,        # (T, 2048) int32
        sm_scale:       cutlass.Constexpr,
        output:         cute.Tensor,        # (T, H, 512)
        lse:            cute.Tensor,        # (T, H)
    ):
        # Block/thread ids
        bidx, bidy, _ = cute.arch.block_idx()    # (T_idx, head_idx)
        tidx, _, _    = cute.arch.thread_idx()
        num_threads: cutlass.Constexpr = BLOCK_SIZE
        num_warps:   cutlass.Constexpr = NUM_WARPS
        wsize                           = cute.arch.WARP_SIZE
        warp_idx  = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx  = cute.arch.lane_idx()

        top_k:         cutlass.Constexpr = TOP_K
        head_dim_ckv:  cutlass.Constexpr = HEAD_DIM_CKV
        head_dim_kpe:  cutlass.Constexpr = HEAD_DIM_KPE
        dims_per_lane: cutlass.Constexpr = DIMS_PER_LANE

        # Shared memory layout
        alloc = cutlass.utils.SmemAllocator()
        smem_logits = alloc.allocate_tensor(
            cutlass.Float32, cute.make_layout((top_k,), stride=(1,)), 16, None)
        smem_sparse = alloc.allocate_tensor(
            cutlass.Int32,   cute.make_layout((top_k,), stride=(1,)),  4, None)
        smem_q_nope = alloc.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((head_dim_ckv,), stride=(1,)), 16, None)
        smem_q_pe = alloc.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((head_dim_kpe,), stride=(1,)), 16, None)
        smem_red_i32 = alloc.allocate_tensor(
            cutlass.Int32,   cute.make_layout((32,), stride=(1,)),  4, None)
        smem_red_f32 = alloc.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,), stride=(1,)), 16, None)
        smem_partial = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((num_warps, head_dim_ckv),
                             stride=(head_dim_ckv, 1)), 16, None)

        # ── Phase 1: Load Q + sparse_indices, count valid ────────────────────
        partial_cnt = 0
        for i in range(tidx, top_k, num_threads):
            idx = sparse_indices[bidx, i]
            smem_sparse[i] = idx
            if idx >= cutlass.Int32(0):
                partial_cnt += 1

        for i in range(tidx, head_dim_ckv, num_threads):
            smem_q_nope[i] = q_nope[bidx, bidy, i]
        for i in range(tidx, head_dim_kpe, num_threads):
            smem_q_pe[i] = q_pe[bidx, bidy, i]

        # Block-wide valid count: warp-reduce then cross-warp
        cnt = _warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = cnt
        cute.arch.sync_threads()

        if warp_idx == 0:
            val = smem_red_i32[lane_idx]
            cnt = _warp_reduce(val, lambda a, b: a + b, width=num_warps)
            smem_red_i32[0] = cnt
        cute.arch.sync_threads()

        valid_count = smem_red_i32[0]
        num_rounds  = (valid_count + num_warps - 1) // num_warps

        # ── Phase 2: SCORE — warp-GEMV dot product ───────────────────────────
        for round_idx in range(num_rounds):
            sparse_idx = round_idx * num_warps + warp_idx
            if sparse_idx < valid_count:
                cur_idx = smem_sparse[sparse_idx]
                s = cutlass.Float32(0)

                # ckv dot: each lane holds 16 dims
                for k in range(head_dim_ckv // wsize):
                    q_n = cutlass.Float32(smem_q_nope[k * wsize + lane_idx])
                    kv  = cutlass.Float32(ckv_cache[cur_idx, k * wsize + lane_idx])
                    s += q_n * kv

                # kpe dot: 64 / 32 = 2 dims per lane
                for k in range(head_dim_kpe // wsize):
                    q_p = cutlass.Float32(smem_q_pe[k * wsize + lane_idx])
                    kp  = cutlass.Float32(kpe_cache[cur_idx, k * wsize + lane_idx])
                    s += q_p * kp

                s_reduced = _warp_reduce(s, lambda a, b: a + b, width=32)
                if lane_idx == 0:
                    smem_logits[sparse_idx] = s_reduced * sm_scale

        cute.arch.sync_threads()

        # ── Phase 3a: Softmax max ─────────────────────────────────────────────
        pmax = -cutlass.Float32(math.inf)
        for i in range(tidx, valid_count, num_threads):
            v = smem_logits[i]
            if v > pmax:
                pmax = v

        m = _warp_reduce(pmax, lambda a, b: a if a > b else b, width=32)
        if lane_idx == 0:
            smem_red_f32[warp_idx] = m
        cute.arch.sync_threads()
        if warp_idx == 0:
            v = smem_red_f32[lane_idx]
            m = _warp_reduce(v, lambda a, b: a if a > b else b, width=num_warps)
            smem_red_f32[0] = m
        cute.arch.sync_threads()
        row_max = smem_red_f32[0]

        # ── Phase 3b: Softmax exp + sum (fused write-back) ────────────────────
        psum = cutlass.Float32(0)
        for i in range(tidx, valid_count, num_threads):
            e = cute.math.exp(smem_logits[i] - row_max)
            smem_logits[i] = e
            psum += e

        ssum = _warp_reduce(psum, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_f32[warp_idx] = ssum
        cute.arch.sync_threads()
        if warp_idx == 0:
            v = smem_red_f32[lane_idx]
            ssum = _warp_reduce(v, lambda a, b: a + b, width=num_warps)
            smem_red_f32[0] = ssum
        cute.arch.sync_threads()
        row_sum = smem_red_f32[0]

        if tidx == 0:
            lse[bidx, bidy] = (row_max + cute.math.log(row_sum)) * cutlass.Float32(LN2_INV)

        # ── Phase 4: OUTPUT — weighted sum of Kc rows (V = K in MLA) ─────────
        # Per-lane register tensor holding 16 output dims
        out_regs = cute.make_rmem_tensor(
            cute.make_layout((dims_per_lane,), stride=(1,)), cutlass.Float32)
        for k in range(dims_per_lane):
            out_regs[k] = cutlass.Float32(0)

        for round_idx in range(num_rounds):
            j = round_idx * num_warps + warp_idx
            if j < valid_count:
                kv_idx = smem_sparse[j]
                w = smem_logits[j] / row_sum       # normalize inline
                for k in range(dims_per_lane):
                    out_regs[k] += w * cutlass.Float32(
                        ckv_cache[kv_idx, k * wsize + lane_idx])

        # Write this warp's partial to smem_partial[warp_idx, :]
        for k in range(dims_per_lane):
            smem_partial[warp_idx, k * wsize + lane_idx] = out_regs[k]
        cute.arch.sync_threads()

        # ── Phase 5: Cross-warp reduce + write bf16 output ───────────────────
        for i in range(tidx, head_dim_ckv, num_threads):
            acc = cutlass.Float32(0)
            for w in range(num_warps):
                acc += smem_partial[w, i]
            output[bidx, bidy, i] = cutlass.BFloat16(acc)


    @cute.jit
    def _entry(
        q_nope:         cute.Tensor,
        q_pe:           cute.Tensor,
        ckv_cache:      cute.Tensor,
        kpe_cache:      cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale:       cutlass.Constexpr,
        output:         cute.Tensor,
        lse:            cute.Tensor,
        stream,
    ):
        T, num_heads, _ = q_nope.shape
        _attn_kernel(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
            sm_scale, output, lse,
        ).launch(
            grid=[T, num_heads, 1],
            block=[BLOCK_SIZE, 1, 1],
            stream=stream,
        )

    # ── Compile at module load ────────────────────────────────────────────────

    def _fake(dtype, shape, stride_order, align):
        return make_fake_compact_tensor(
            dtype=dtype, shape=shape,
            stride_order=stride_order, assumed_align=align,
        )

    def _compile():
        T_sym = cute.sym_int()
        N_sym = cute.sym_int()
        q_nope = _fake(cute.BFloat16, (T_sym, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
        q_pe   = _fake(cute.BFloat16, (T_sym, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0), 16)
        ckv    = _fake(cute.BFloat16, (N_sym, HEAD_DIM_CKV), (1, 0), 16)
        kpe    = _fake(cute.BFloat16, (N_sym, HEAD_DIM_KPE), (1, 0), 16)
        idx    = _fake(cute.Int32,    (T_sym, TOP_K), (1, 0), 4)
        output = _fake(cute.BFloat16, (T_sym, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
        lse    = _fake(cute.Float32,  (T_sym, NUM_HEADS), (1, 0), 4)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)

        return cute.compile(
            _entry,
            q_nope, q_pe, ckv, kpe, idx, 0.1352337788608801, output, lse, stream,
            options="--enable-tvm-ffi",
        )

    try:
        _compiled = _compile()
        print("[kernel_cute] iter1: compiled OK", flush=True)
    except Exception:
        print("[kernel_cute] cute.compile FAILED:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# Public DPS entry
# ═══════════════════════════════════════════════════════════════════════════════

def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        sm_scale, output, lse):
    """DPS entry. Matches reference kernel2..kernel5cl.py signature."""
    if not CUTLASS_AVAILABLE:
        raise RuntimeError(
            f"cutlass.cute not available (import failed: {_import_err})."
        )
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    _compiled(q_nope, q_pe, ckv_flat, kpe_flat,
              sparse_indices, output, lse)


# Alias for dev tools
kernel = run
