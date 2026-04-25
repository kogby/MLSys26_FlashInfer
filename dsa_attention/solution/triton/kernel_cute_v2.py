"""
CUTLASS cute port — iter2: KV-split (compute + reduce), grid [T, H, NUM_SPLITS].

Why: iter1 processed all 2048 KV indices sequentially per block → large-T
latencies crept up to 0.13ms. Splitting into 8 chunks of 256 gives 8x more
parallel blocks (for T=8: 8*16*8=1024 blocks vs iter1's 128).

Design:
  _compute_kernel  grid [T, H, NUM_SPLITS] : per-split attention, 256 indices.
                   Writes partial_out[T,H,S,512] + partial_lse[T,H,S,2]=(m,l).
  _reduce_kernel   grid [T, H]             : LSE-merge across splits, writes
                   final output + lse.

Softmax: 2-pass (same as iter1 inside each split). A later iter may convert to
single-pass online softmax.

Workspace: partial_out/partial_lse allocated once, cached by (T, device).
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

NUM_SPLITS    = 8
DIM_SPLIT     = TOP_K // NUM_SPLITS         # 256

# Compute kernel
BLOCK_SIZE    = 1024
NUM_WARPS     = 32
WARP_SIZE     = 32
DIMS_PER_LANE = HEAD_DIM_CKV // WARP_SIZE   # 16

# Reduce kernel — one thread per output dim is enough (512 dims → 512 threads max;
# use 128 threads looping).
REDUCE_BLOCK_SIZE = 128

LN2_INV = 1.4426950408889634


# ═══════════════════════════════════════════════════════════════════════════════
# cute kernels
# ═══════════════════════════════════════════════════════════════════════════════

if CUTLASS_AVAILABLE:

    @cute.jit
    def _warp_reduce(val: cute.Numeric, op: callable,
                     width: cutlass.Constexpr = 32) -> cute.Numeric:
        for i in range(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val

    # ───────────────────────────────────────────────────────────────────────────
    # Compute kernel: one (T, head, split) per block
    # ───────────────────────────────────────────────────────────────────────────

    @cute.kernel
    def _compute_kernel(
        q_nope:         cute.Tensor,   # (T, H, 512)
        q_pe:           cute.Tensor,   # (T, H, 64)
        ckv_cache:      cute.Tensor,   # (N, 512) flat
        kpe_cache:      cute.Tensor,   # (N, 64)  flat
        sparse_indices: cute.Tensor,   # (T, 2048) int32
        sm_scale:       cutlass.Constexpr,
        partial_out:    cute.Tensor,   # (T, H, S, 512) f32
        partial_lse:    cute.Tensor,   # (T, H, S, 2)   f32   (0=m, 1=l)
    ):
        bidx, bidy, bidz = cute.arch.block_idx()   # (T_idx, head_idx, split_idx)
        tidx, _, _       = cute.arch.thread_idx()
        num_threads: cutlass.Constexpr = BLOCK_SIZE
        num_warps:   cutlass.Constexpr = NUM_WARPS
        wsize                           = cute.arch.WARP_SIZE
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        dim_split:     cutlass.Constexpr = DIM_SPLIT
        top_k:         cutlass.Constexpr = TOP_K
        head_dim_ckv:  cutlass.Constexpr = HEAD_DIM_CKV
        head_dim_kpe:  cutlass.Constexpr = HEAD_DIM_KPE
        dims_per_lane: cutlass.Constexpr = DIMS_PER_LANE

        split_start = bidz * dim_split   # this split's offset into sparse_indices

        # Shared memory layout — logits is only dim_split=256, much smaller than iter1.
        alloc = cutlass.utils.SmemAllocator()
        smem_logits = alloc.allocate_tensor(
            cutlass.Float32, cute.make_layout((dim_split,), stride=(1,)), 16, None)
        smem_sparse = alloc.allocate_tensor(
            cutlass.Int32,   cute.make_layout((dim_split,), stride=(1,)),  4, None)
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

        # ── Phase 1: Load Q + this split's sparse_indices, count valid in split ──
        partial_cnt = 0
        for i in range(tidx, dim_split, num_threads):
            idx = sparse_indices[bidx, split_start + i]
            smem_sparse[i] = idx
            if idx >= cutlass.Int32(0):
                partial_cnt += 1

        for i in range(tidx, head_dim_ckv, num_threads):
            smem_q_nope[i] = q_nope[bidx, bidy, i]
        for i in range(tidx, head_dim_kpe, num_threads):
            smem_q_pe[i] = q_pe[bidx, bidy, i]

        # Block-wide valid count
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
        # No early return allowed in cute — handle empty split in the final
        # write branch below.  Compute phases below all guard by `sparse_idx <
        # valid_count`, so an empty split naturally does no work and leaves
        # out_regs at zero.
        num_rounds = (valid_count + num_warps - 1) // num_warps

        # ── Phase 2: SCORE (warp-GEMV, same as iter1) ────────────────────────
        for round_idx in range(num_rounds):
            sparse_idx = round_idx * num_warps + warp_idx
            if sparse_idx < valid_count:
                cur_idx = smem_sparse[sparse_idx]
                s = cutlass.Float32(0)

                for k in range(head_dim_ckv // wsize):
                    q_n = cutlass.Float32(smem_q_nope[k * wsize + lane_idx])
                    kv  = cutlass.Float32(ckv_cache[cur_idx, k * wsize + lane_idx])
                    s += q_n * kv

                for k in range(head_dim_kpe // wsize):
                    q_p = cutlass.Float32(smem_q_pe[k * wsize + lane_idx])
                    kp  = cutlass.Float32(kpe_cache[cur_idx, k * wsize + lane_idx])
                    s += q_p * kp

                s_reduced = _warp_reduce(s, lambda a, b: a + b, width=32)
                if lane_idx == 0:
                    smem_logits[sparse_idx] = s_reduced * sm_scale

        cute.arch.sync_threads()

        # ── Phase 3a: softmax max ────────────────────────────────────────────
        # Use -1e30 sentinel (not -inf) so downstream exp(-inf - (-inf)) = NaN
        # is avoided for empty splits (same trick as V4 triton).
        pmax = cutlass.Float32(-1.0e30)
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

        # ── Phase 3b: softmax exp + sum (fused) ──────────────────────────────
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

        # ── Phase 4: OUTPUT — weighted sum of V=K rows ──────────────────────
        # Write UNNORMALIZED acc (weights not divided by row_sum).
        # Reduce kernel normalizes after merging across splits.
        out_regs = cute.make_rmem_tensor(
            cute.make_layout((dims_per_lane,), stride=(1,)), cutlass.Float32)
        for k in range(dims_per_lane):
            out_regs[k] = cutlass.Float32(0)

        for round_idx in range(num_rounds):
            j = round_idx * num_warps + warp_idx
            if j < valid_count:
                kv_idx = smem_sparse[j]
                w      = smem_logits[j]  # NOT divided by row_sum — reduce does it
                for k in range(dims_per_lane):
                    out_regs[k] += w * cutlass.Float32(
                        ckv_cache[kv_idx, k * wsize + lane_idx])

        for k in range(dims_per_lane):
            smem_partial[warp_idx, k * wsize + lane_idx] = out_regs[k]
        cute.arch.sync_threads()

        # ── Phase 5: cross-warp reduce + write partial_out + (m, l) ──────────
        for i in range(tidx, head_dim_ckv, num_threads):
            acc = cutlass.Float32(0)
            for w in range(num_warps):
                acc += smem_partial[w, i]
            partial_out[bidx, bidy, bidz, i] = acc

        if tidx == 0:
            partial_lse[bidx, bidy, bidz, 0] = row_max
            partial_lse[bidx, bidy, bidz, 1] = row_sum

    # ───────────────────────────────────────────────────────────────────────────
    # Reduce kernel: one (T, head) per block, merges NUM_SPLITS partials
    # ───────────────────────────────────────────────────────────────────────────

    @cute.kernel
    def _reduce_kernel(
        partial_out: cute.Tensor,   # (T, H, S, 512) f32
        partial_lse: cute.Tensor,   # (T, H, S, 2)   f32
        output:      cute.Tensor,   # (T, H, 512)    bf16
        lse:         cute.Tensor,   # (T, H)         f32
    ):
        bidx, bidy, _ = cute.arch.block_idx()   # (T_idx, head_idx)
        tidx, _, _    = cute.arch.thread_idx()
        num_threads:  cutlass.Constexpr = REDUCE_BLOCK_SIZE
        num_splits:   cutlass.Constexpr = NUM_SPLITS
        head_dim_ckv: cutlass.Constexpr = HEAD_DIM_CKV

        # Each thread redundantly computes g_max and g_sum — cheap (num_splits=8)
        # and avoids smem broadcast sync.

        g_max = cutlass.Float32(-1.0e30)
        for s in range(num_splits):
            v = partial_lse[bidx, bidy, s, 0]
            if v > g_max:
                g_max = v

        g_sum = cutlass.Float32(0)
        for s in range(num_splits):
            l_s = partial_lse[bidx, bidy, s, 1]
            m_s = partial_lse[bidx, bidy, s, 0]
            g_sum += l_s * cute.math.exp(m_s - g_max)

        # Write lse (guarded so all-empty case writes -inf cleanly)
        if tidx == 0:
            if g_sum > cutlass.Float32(0):
                lse[bidx, bidy] = (g_max + cute.math.log(g_sum)) * cutlass.Float32(LN2_INV)
            else:
                lse[bidx, bidy] = cutlass.Float32(-math.inf)

        # Guard division when all splits empty (g_sum == 0).
        safe_sum = g_sum
        if g_sum <= cutlass.Float32(0):
            safe_sum = cutlass.Float32(1e-30)

        # Merge: out[d] = sum_s( exp(m_s - g_max) * partial_out[s, d] ) / g_sum
        for d in range(tidx, head_dim_ckv, num_threads):
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                m_s   = partial_lse[bidx, bidy, s, 0]
                scale = cute.math.exp(m_s - g_max)
                acc  += scale * partial_out[bidx, bidy, s, d]
            output[bidx, bidy, d] = cutlass.BFloat16(acc / safe_sum)

    # ───────────────────────────────────────────────────────────────────────────
    # JIT entry — launches compute + reduce
    # ───────────────────────────────────────────────────────────────────────────

    @cute.jit
    def _entry(
        q_nope:         cute.Tensor,
        q_pe:           cute.Tensor,
        ckv_cache:      cute.Tensor,
        kpe_cache:      cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale:       cutlass.Constexpr,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor,
        stream,
    ):
        T, num_heads, _ = q_nope.shape
        _compute_kernel(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
            sm_scale, partial_out, partial_lse,
        ).launch(
            grid=[T, num_heads, NUM_SPLITS],
            block=[BLOCK_SIZE, 1, 1],
            stream=stream,
        )
        _reduce_kernel(
            partial_out, partial_lse, output, lse,
        ).launch(
            grid=[T, num_heads, 1],
            block=[REDUCE_BLOCK_SIZE, 1, 1],
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
        q_nope   = _fake(cute.BFloat16, (T_sym, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
        q_pe     = _fake(cute.BFloat16, (T_sym, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0), 16)
        ckv      = _fake(cute.BFloat16, (N_sym, HEAD_DIM_CKV), (1, 0), 16)
        kpe      = _fake(cute.BFloat16, (N_sym, HEAD_DIM_KPE), (1, 0), 16)
        idx      = _fake(cute.Int32,    (T_sym, TOP_K), (1, 0), 4)
        part_out = _fake(cute.Float32,  (T_sym, NUM_HEADS, NUM_SPLITS, HEAD_DIM_CKV),
                         (3, 2, 1, 0), 16)
        part_lse = _fake(cute.Float32,  (T_sym, NUM_HEADS, NUM_SPLITS, 2),
                         (3, 2, 1, 0), 16)
        output   = _fake(cute.BFloat16, (T_sym, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
        lse      = _fake(cute.Float32,  (T_sym, NUM_HEADS), (1, 0), 4)
        stream   = make_fake_stream(use_tvm_ffi_env_stream=True)

        return cute.compile(
            _entry,
            q_nope, q_pe, ckv, kpe, idx, 0.1352337788608801,
            part_out, part_lse, output, lse, stream,
            options="--enable-tvm-ffi",
        )

    try:
        _compiled = _compile()
        print("[kernel_cute] iter2: compiled OK", flush=True)
    except Exception:
        print("[kernel_cute] cute.compile FAILED:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# Workspace caching (partial_out, partial_lse)
# ═══════════════════════════════════════════════════════════════════════════════

_WORKSPACE = {}

def _get_workspace(T, device):
    key = (T, str(device))
    ws = _WORKSPACE.get(key)
    if ws is None:
        partial_out = torch.empty(
            T, NUM_HEADS, NUM_SPLITS, HEAD_DIM_CKV,
            dtype=torch.float32, device=device,
        )
        partial_lse = torch.empty(
            T, NUM_HEADS, NUM_SPLITS, 2,
            dtype=torch.float32, device=device,
        )
        ws = (partial_out, partial_lse)
        _WORKSPACE[key] = ws
    return ws


# ═══════════════════════════════════════════════════════════════════════════════
# Public DPS entry
# ═══════════════════════════════════════════════════════════════════════════════

def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        sm_scale, output, lse):
    if not CUTLASS_AVAILABLE:
        raise RuntimeError(
            f"cutlass.cute not available (import failed: {_import_err})."
        )
    T = q_nope.shape[0]
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    partial_out, partial_lse = _get_workspace(T, q_nope.device)
    _compiled(
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
        partial_out, partial_lse, output, lse,
    )


# Alias for dev tools
kernel = run
