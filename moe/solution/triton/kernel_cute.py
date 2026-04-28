"""
Fused MoE FP8 kernel — CuTe DSL GEMM2 variant (Phase A skeleton).

This file mirrors v11 (kernel.py) but routes GEMM2 through a CuTe DSL kernel
when available. The CuTe path is currently a stub — the wrapper falls back to
v11's Triton _grouped_gemm2 unconditionally until the @cute.kernel is filled
in. This keeps the file submittable end-to-end while the CuTe implementation
is built up incrementally.

Phase A goal: fp32×FP8 → fp32 grouped GEMM2 in CuTe DSL (bf16 tcgen05 MMA after
in-smem cast). See plan at ~/.claude/plans/cute-gemm2-happy-crown.md.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

import torch
import triton

# ─────────────────────────────────────────────────────────────────────────────
# Sibling import of v11 kernel.py
# ─────────────────────────────────────────────────────────────────────────────
# spec_from_file_location works whether the harness imports as a package or
# loads this file standalone — same trick as dsa_attention/kernel_cute_hybrid.
_HERE = Path(__file__).resolve().parent


def _load_sibling(name: str, fname: str):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, _HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_v11 = _load_sibling("_v11_kernel", "kernel.py")

# Re-export constants from v11 to keep this file self-contained.
H = _v11.H
I = _v11.I
E_LOCAL = _v11.E_LOCAL
TOP_K = _v11.TOP_K

# ─────────────────────────────────────────────────────────────────────────────
# CuTe TF32 grouped GEMM2 (Phase A)
#
# A: fp32 [total_tokens, I=2048] with stride (2*I, 1) (gemm1_out, first I cols)
# B: fp8 e4m3 [E_LOCAL=32, H=7168, I=2048]
# SB: fp32 [E_LOCAL, H/128, I/128] block scales (per-(N,K)-tile scalar)
# EO: int32 [E_LOCAL+1] expert offsets in M dim
# C: fp32 [total_tokens, H=7168] output
#
# Compute path: fp8 B + fp32 SB → dequant in rmem to fp32 → cast to TF32 in smem.
# A: fp32 → cast to TF32 in smem. tcgen05.MmaTF32Op consumes TF32×TF32 → fp32.
#
# Phase A goal: correctness + ≥ v11 latency. Pipelining/PDL deferred.
# ─────────────────────────────────────────────────────────────────────────────

_GEMM2_CUTE = None
_GEMM2_COMPILED = None

# Tile shape — fixed for Phase A. tcgen05 UMMA requires M ∈ {64,128,256}, N ∈ {128,256}.
_BLOCK_M_CUTE = 128
_BLOCK_N_CUTE = 128
_BLOCK_K_CUTE = 128  # matches SB block size, so one SB load per K-iteration

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cute_utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from cutlass.cute.nvgpu import tcgen05, cpasync

    _MMA_INST_MNK = (_BLOCK_M_CUTE, _BLOCK_N_CUTE, 8)  # tf32 MMA inst K=8
    _CTA_TILE_MNK = (_BLOCK_M_CUTE, _BLOCK_N_CUTE, _BLOCK_K_CUTE)
    _THREADS_PER_CTA = 128

    class GEMM2_CuTe:
        def __init__(self):
            self.ab_dtype = cutlass.TFloat32
            self.acc_dtype = cutlass.Float32

        @cute.jit
        def __call__(self, A, B, SB, EO, C, max_M_tiles, stream):
            """Host-side launch entry. A/B/SB/EO/C are full-shape tensors;
            max_M_tiles is the max ceil(M_e / BLOCK_M) across local experts.
            """
            # MmaTF32Op signature (cutlass-dsl 4.4.2):
            #   (instruction_shape, cta_group, a_src, a_major_mode, b_major_mode)
            # ab_dtype (TF32) and acc_dtype (F32) are baked in.
            op = tcgen05.MmaTF32Op(
                _MMA_INST_MNK, tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            )
            tiled_mma = cute.make_tiled_mma(op)

            a_smem_layout = sm100_utils.make_smem_layout_a(
                tiled_mma, _CTA_TILE_MNK, self.ab_dtype, num_stages=1,
            )
            b_smem_layout = sm100_utils.make_smem_layout_b(
                tiled_mma, _CTA_TILE_MNK, self.ab_dtype, num_stages=1,
            )

            @cute.struct
            class SharedStorage:
                mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
                tmem_holding_buf: cutlass.Int32
            self.shared_storage = SharedStorage

            # Grid: (max_M_tiles, H/BLOCK_N, E_LOCAL)
            n_tiles = H // _BLOCK_N_CUTE  # 7168 / 128 = 56
            self._gemm2_kernel(
                tiled_mma, a_smem_layout, b_smem_layout,
                A, B, SB, EO, C,
            ).launch(
                grid=[max_M_tiles, n_tiles, E_LOCAL],
                block=[_THREADS_PER_CTA, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def _gemm2_kernel(
            self, tiled_mma, a_smem_layout, b_smem_layout,
            A, B, SB, EO, C,
        ):
            ab_dtype: cutlass.Constexpr = self.ab_dtype
            acc_dtype: cutlass.Constexpr = self.acc_dtype

            tidx, _, _ = cute.arch.thread_idx()
            pid_m, pid_n, pid_e = cute.arch.block_idx()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

            # ── per-expert M range ────────────────────────────────────────────
            # CuTe DSL forbids dynamic early-return inside @cute.kernel; instead
            # wrap the whole body in `if m_local_start < M_e`. Out-of-range CTAs
            # do nothing (no smem alloc, no MMA, no store) but still walk the
            # function epilogue.
            e_start = EO[pid_e]
            e_end = EO[pid_e + 1]
            M_e = e_end - e_start
            m_local_start = pid_m * cutlass.Int32(_BLOCK_M_CUTE)

            if m_local_start < M_e:
                # ── SMEM allocation ───────────────────────────────────────────
                alloc = cute_utils.SmemAllocator()
                sA = alloc.allocate_tensor(
                    element_type=ab_dtype, layout=a_smem_layout.outer,
                    byte_alignment=128, swizzle=a_smem_layout.inner,
                )
                sB = alloc.allocate_tensor(
                    element_type=ab_dtype, layout=b_smem_layout.outer,
                    byte_alignment=128, swizzle=b_smem_layout.inner,
                )
                storage = alloc.allocate(self.shared_storage)
                mma_mbar = storage.mma_mbar_ptr.data_ptr()

                # ── TMEM accumulator setup ────────────────────────────────────
                acc_shape = tiled_mma.partition_shape_C(_CTA_TILE_MNK[:2])
                tCtAcc_tmpl = tiled_mma.make_fragment_C(acc_shape)
                num_tmem_cols = cute_utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
                if warp_idx == 0:
                    cute.arch.alloc_tmem(
                        cutlass.Int32(num_tmem_cols), storage.tmem_holding_buf,
                    )
                cute.arch.barrier(barrier_id=1, number_of_threads=_THREADS_PER_CTA)
                tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    acc_dtype, alignment=16,
                    ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
                )
                tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)

                # ── mbar init ────────────────────────────────────────────────
                if warp_idx == 0:
                    if tidx == 0:
                        cute.arch.mbarrier_init(mma_mbar, cnt=1)
                        cute.arch.mbarrier_init_fence()
                cute.arch.barrier(barrier_id=1, number_of_threads=_THREADS_PER_CTA)

                tCrA = tiled_mma.make_fragment_A(sA)
                tCrB = tiled_mma.make_fragment_B(sB)
                num_k_blocks = cute.size(tCrA, mode=[2])

                # ── K-loop: for each BLOCK_K tile in I=2048 dim ──────────────
                mma_phase = cutlass.Int32(0)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                K_DIM: cutlass.Constexpr = I
                num_k_tiles: cutlass.Constexpr = K_DIM // _BLOCK_K_CUTE  # 16

                for kb in cutlass.range_constexpr(num_k_tiles):
                    k_start = kb * cutlass.Int32(_BLOCK_K_CUTE)

                    # ── Load A tile (fp32) → cast TF32 → smem ─────────────────
                    ELEMS_PER_THREAD_A: cutlass.Constexpr = (
                        _BLOCK_M_CUTE * _BLOCK_K_CUTE // _THREADS_PER_CTA
                    )
                    for i in cutlass.range_constexpr(ELEMS_PER_THREAD_A):
                        flat_idx = tidx * cutlass.Int32(ELEMS_PER_THREAD_A) + cutlass.Int32(i)
                        a_m = flat_idx // cutlass.Int32(_BLOCK_K_CUTE)
                        a_k = flat_idx - a_m * cutlass.Int32(_BLOCK_K_CUTE)
                        g_m = e_start + m_local_start + a_m
                        g_k = k_start + a_k
                        a_val = cutlass.Float32(0.0)
                        if (e_start + m_local_start + a_m) < e_end:
                            a_val = A[g_m, g_k]
                        sA[a_m, a_k] = ab_dtype(a_val)

                    # ── Load B tile (fp8) → dequant + scale → cast TF32 → smem
                    n_start = pid_n * cutlass.Int32(_BLOCK_N_CUTE)
                    sb_scale = SB[pid_e, pid_n, cutlass.Int32(kb)]
                    ELEMS_PER_THREAD_B: cutlass.Constexpr = (
                        _BLOCK_N_CUTE * _BLOCK_K_CUTE // _THREADS_PER_CTA
                    )
                    for i in cutlass.range_constexpr(ELEMS_PER_THREAD_B):
                        flat_idx = tidx * cutlass.Int32(ELEMS_PER_THREAD_B) + cutlass.Int32(i)
                        b_n = flat_idx // cutlass.Int32(_BLOCK_K_CUTE)
                        b_k = flat_idx - b_n * cutlass.Int32(_BLOCK_K_CUTE)
                        fp8_val = B[pid_e, n_start + b_n, k_start + b_k]
                        f32_val = cutlass.Float32(fp8_val) * sb_scale
                        sB[b_n, b_k] = ab_dtype(f32_val)

                    cute.arch.sync_threads()
                    cute.arch.fence_view_async_shared()

                    # ── tcgen05 TF32 MMA (warp 0 only) ────────────────────────
                    if warp_idx == 0:
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_coord = (None, None, k_block_idx, 0)
                            cute.gemm(
                                tiled_mma, tCtAcc,
                                tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        if tidx == 0:
                            tcgen05.commit(mma_mbar)
                    cute.arch.mbarrier_wait(mma_mbar, mma_phase)
                    mma_phase = mma_phase ^ cutlass.Int32(1)
                    cute.arch.sync_threads()

                # ── Epilogue: TMEM → rmem fp32 → gmem with M_e mask ──────────
                M_acc = cute.size(tCtAcc, mode=[0, 0])
                ld_op = tcgen05.Ld32x32bOp(tcgen05.Repetition(_BLOCK_N_CUTE))
                epi_tiler = ((M_acc, _BLOCK_N_CUTE),)
                tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
                copy_atom_t2r = cute.make_copy_atom(ld_op, acc_dtype)
                tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
                tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
                tTR_tAcc = tmem_thr_copy.partition_S(tCtAcc_epi)
                tTR_rAcc = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, acc_dtype)

                if tidx < cutlass.Int32(_BLOCK_M_CUTE):
                    cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

                    m_local = m_local_start + tidx
                    g_m = e_start + m_local
                    if m_local < M_e:
                        for n_idx in cutlass.range_constexpr(_BLOCK_N_CUTE):
                            n_global = pid_n * cutlass.Int32(_BLOCK_N_CUTE) + cutlass.Int32(n_idx)
                            C[g_m, n_global] = tTR_rAcc[n_idx]

                cute.arch.sync_threads()

    def _compile_gemm2(instance):
        T_ = cute.sym_int()
        max_M_tiles_ = cute.sym_int()

        # A is fp32 [T, I]. The actual stride at runtime is (2*I, 1) but we
        # declare with default contiguous stride; the runtime tensor passes its
        # actual strides through CuTe's tensor descriptor mechanism.
        A_fake = make_fake_compact_tensor(
            dtype=cutlass.Float32, shape=(T_, I), stride_order=(1, 0),
            assumed_align=4,
        )
        B_fake = make_fake_compact_tensor(
            dtype=cutlass.Float8E4M3FN, shape=(E_LOCAL, H, I),
            stride_order=(2, 1, 0), assumed_align=16,
        )
        SB_fake = make_fake_compact_tensor(
            dtype=cutlass.Float32, shape=(E_LOCAL, H // 128, I // 128),
            stride_order=(2, 1, 0), assumed_align=4,
        )
        EO_fake = make_fake_compact_tensor(
            dtype=cutlass.Int32, shape=(E_LOCAL + 1,),
            stride_order=(0,), assumed_align=4,
        )
        C_fake = make_fake_compact_tensor(
            dtype=cutlass.Float32, shape=(T_, H), stride_order=(1, 0),
            assumed_align=4,
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)

        return cute.compile(
            instance,
            A_fake, B_fake, SB_fake, EO_fake, C_fake, max_M_tiles_, stream,
            options="--enable-tvm-ffi",
        )

    # NO fallback — let cute.compile failures propagate so they show up in
    # the evaluator's per-workload log.
    _GEMM2_CUTE = GEMM2_CuTe()
    _GEMM2_COMPILED = _compile_gemm2(_GEMM2_CUTE)
    print("[kernel_cute] CuTe GEMM2 compiled OK", flush=True)

except Exception:
    print("[kernel_cute] CuTe setup failed — see traceback below", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    raise

# ─────────────────────────────────────────────────────────────────────────────
# FIB_PROFILE: per-stage CUDA event timing (gated)
# ─────────────────────────────────────────────────────────────────────────────
_PROFILE = os.environ.get("FIB_PROFILE", "0") == "1"


class _StageTimer:
    """Lightweight per-call CUDA-event timer. Disabled unless FIB_PROFILE=1."""

    def __init__(self):
        self.events = []  # list[(label, start_event, end_event)]
        self.cur = None

    def start(self, label: str):
        if not _PROFILE:
            return
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        self.cur = (label, s, e)

    def stop(self):
        if not _PROFILE or self.cur is None:
            return
        label, s, e = self.cur
        e.record()
        self.events.append((label, s, e))
        self.cur = None

    def report(self):
        if not _PROFILE or not self.events:
            return
        torch.cuda.synchronize()
        line = "  ".join(
            f"{lbl}={s.elapsed_time(e):.3f}ms" for lbl, s, e in self.events
        )
        print(f"[kernel_cute stages] {line}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def kernel(
    routing_logits:        torch.Tensor,
    routing_bias:          torch.Tensor,
    hidden_states:         torch.Tensor,
    hidden_states_scale:   torch.Tensor,
    gemm1_weights:         torch.Tensor,
    gemm1_weights_scale:   torch.Tensor,
    gemm2_weights:         torch.Tensor,
    gemm2_weights_scale:   torch.Tensor,
    local_expert_offset:   int,
    routed_scaling_factor: float,
    output:                torch.Tensor,
):
    T = hidden_states.shape[0]
    device = hidden_states.device
    local_start = int(local_expert_offset)
    timer = _StageTimer()

    # ── 1. Fused routing ─────────────────────────────────────────────────────
    timer.start("routing")
    topk_idx = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
    topk_weights = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
    _v11._routing_kernel[(T,)](
        routing_logits, routing_bias, topk_idx, topk_weights, T, routed_scaling_factor,
    )
    timer.stop()

    # ── 2. Count tokens per local expert ─────────────────────────────────────
    timer.start("count")
    expert_counts = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    _v11._count_expert_tokens[(T,)](
        topk_idx, expert_counts, local_start, T,
    )
    timer.stop()

    # ── 3. GPU cumsum + 1 CPU sync ───────────────────────────────────────────
    timer.start("offsets")
    expert_offsets_gpu = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=device)
    expert_offsets_gpu[1:] = expert_counts.cumsum(0).to(torch.int32)
    expert_offsets_cpu = expert_offsets_gpu.cpu()
    total_tokens = int(expert_offsets_cpu[-1])
    token_counts_cpu = (expert_offsets_cpu[1:] - expert_offsets_cpu[:-1]).tolist()
    timer.stop()

    if total_tokens == 0:
        output.zero_()
        timer.report()
        return

    # ── 4. Scatter tokens into sorted layout ─────────────────────────────────
    timer.start("scatter")
    sorted_token_ids = torch.empty(total_tokens, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(total_tokens, dtype=torch.float32, device=device)
    write_ptrs = expert_offsets_gpu[:-1].clone()
    _v11._scatter_sorted_tokens[(T,)](
        topk_idx, topk_weights, write_ptrs,
        sorted_token_ids, sorted_weights,
        local_start, T,
    )
    timer.stop()

    # ── 5. Gather sorted hidden states ───────────────────────────────────────
    timer.start("gather")
    sorted_A = hidden_states[sorted_token_ids].contiguous()
    sorted_A_scale = hidden_states_scale[:, sorted_token_ids].contiguous()
    timer.stop()

    # ── 6. Grouped GEMM1 (FP8 × FP8 → fp32) ──────────────────────────────────
    timer.start("gemm1")
    N1 = 2 * I
    gemm1_out = torch.empty((total_tokens, N1), dtype=torch.float32, device=device)
    grid1 = lambda meta: (
        sum((int(tc) + meta['BLOCK_M'] - 1) // meta['BLOCK_M'] for tc in token_counts_cpu),
        triton.cdiv(N1, meta['BLOCK_N']),
    )
    _v11._grouped_gemm1_swiglu[grid1](
        sorted_A, sorted_A_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm1_out,
        expert_offsets_gpu,
        N1, H, total_tokens,
        sorted_A.stride(0), sorted_A.stride(1),
        sorted_A_scale.stride(0), sorted_A_scale.stride(1),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        gemm1_out.stride(0), gemm1_out.stride(1),
    )
    timer.stop()

    # ── 7. SwiGLU in-place (writes silu(up)*gate into first I cols) ──────────
    timer.start("swiglu")
    _v11._swiglu_inplace[
        (triton.cdiv(total_tokens, 32), triton.cdiv(I, 128))
    ](gemm1_out, total_tokens, I=I, BLOCK_M=32, BLOCK_I=128)
    timer.stop()

    # ── 8. Grouped GEMM2 (CuTe tcgen05 path) ─────────────────────────────────
    # Sanity guard: GEMM2 reads A from gemm1_out's first I cols using physical
    # stride 2*I.
    assert gemm1_out.stride() == (2 * I, 1), (
        f"Unexpected gemm1_out stride: {gemm1_out.stride()}, expected ({2*I}, 1)"
    )
    gemm2_out = torch.empty((total_tokens, H), dtype=torch.float32, device=device)

    timer.start("gemm2")
    # No fallback: if cute.compile failed, module-load already raised.
    max_M_e = max(token_counts_cpu) if token_counts_cpu else 0
    max_M_tiles = max((max_M_e + _BLOCK_M_CUTE - 1) // _BLOCK_M_CUTE, 1)
    _GEMM2_COMPILED(
        gemm1_out, gemm2_weights, gemm2_weights_scale,
        expert_offsets_gpu, gemm2_out,
        max_M_tiles, torch.cuda.current_stream(),
    )
    timer.stop()

    # ── 9. Weighted scatter + cast ───────────────────────────────────────────
    timer.start("scatter_out")
    weighted = gemm2_out * sorted_weights.unsqueeze(1)
    out_f32 = torch.zeros((T, H), dtype=torch.float32, device=device)
    out_f32.index_add_(0, sorted_token_ids.long(), weighted)
    output.copy_(out_f32)
    timer.stop()

    timer.report()
