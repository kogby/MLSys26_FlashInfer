"""
H100 (sm_90a) wgmma TF32 grouped GEMM2 — DEV-ONLY variant of kernel_cute.py.

Submission target is B200 (sm_100a, tcgen05 path in kernel_cute.py). This file
exists so we can iterate on the CuTe DSL pipeline (sibling import, fake tensor
strides, scalar gmem indexing, smem layout helpers, MMA wiring) on H100 while
B200 instances are queued. Once H100 verifies the assumptions, port the deltas
back to kernel_cute.py.

Key wgmma-vs-tcgen05 deltas:
  - warpgroup.MmaOp(TFloat32, TFloat32, Float32, ...) instead of MmaTF32Op
  - acc lives in registers via make_fragment_C (no TMEM, no alloc_tmem,
    no retrieve_tmem_ptr, no Ld32x32bOp epilogue)
  - sync via wgmma.fence + commit_group + wait_group(0) instead of mbarrier
  - whole warpgroup (128 threads) cooperatively executes MMA — no warp-0 gate
  - BLOCK_M = 64 (wgmma instruction is M=64 fixed)
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


# ─────────────────────────────────────────────────────────────────────────────
# Local Triton helper: SwiGLU + per-row (full-K) FP16 quantization.
# Simplified vs v15 _swiglu_to_fp16_scaled: one scale per row instead of per
# (row, 128-K-block). Trades some FP16 dynamic range for a much simpler CuTe
# epilogue (single scalar multiply instead of per-K-tile multiply).
# ─────────────────────────────────────────────────────────────────────────────

import triton.language as tl  # noqa: E402


@triton.jit
def _swiglu_to_fp16_perrow(
    X_ptr,           # [total_tokens, 2*I] FP32 (gemm1_out)
    Y_ptr,           # [total_tokens, I]   FP16 (output)
    SA_ptr,          # [total_tokens]      FP32 (per-row scale)
    total_tokens,
    I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < total_tokens

    # First pass: compute row_max across the full I dimension.
    row_max = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for i_start in range(0, I, BLOCK_I):
        ri = i_start + tl.arange(0, BLOCK_I)
        mask = mask_m[:, None] & (ri[None, :] < I)
        gate = tl.load(X_ptr + rm[:, None] * (2 * I) + ri[None, :],
                       mask=mask, other=0.0)
        up = tl.load(X_ptr + rm[:, None] * (2 * I) + (ri[None, :] + I),
                     mask=mask, other=0.0)
        z = tl.sigmoid(up) * up * gate
        row_max = tl.maximum(row_max, tl.max(tl.abs(z), axis=1))

    scale = tl.maximum(row_max, 1e-30) / 32000.0  # target |z_fp16| ≤ 32000
    tl.store(SA_ptr + rm, scale, mask=mask_m)

    # Second pass: write z / scale as FP16.
    for i_start in range(0, I, BLOCK_I):
        ri = i_start + tl.arange(0, BLOCK_I)
        mask = mask_m[:, None] & (ri[None, :] < I)
        gate = tl.load(X_ptr + rm[:, None] * (2 * I) + ri[None, :],
                       mask=mask, other=0.0)
        up = tl.load(X_ptr + rm[:, None] * (2 * I) + (ri[None, :] + I),
                     mask=mask, other=0.0)
        z = tl.sigmoid(up) * up * gate
        z_fp16 = (z / scale[:, None]).to(tl.float16)
        tl.store(Y_ptr + rm[:, None] * I + ri[None, :], z_fp16, mask=mask)

H = _v11.H
I = _v11.I
E_LOCAL = _v11.E_LOCAL
TOP_K = _v11.TOP_K

# ─────────────────────────────────────────────────────────────────────────────
# Pre-dequant cache for gemm2_weights (Path B / sm_90 wgmma F16 path)
#
# wgmma + sm_90 has a vector restriction on nvgpu.cvt_fpext that blocks
# scalar fp8→f16 cast inside the kernel. We sidestep by materializing a fp16
# copy of B = (gemm2_weights_fp8 * SB) once per weight tensor (weights don't
# change across calls), and pass the fp16 tensor to the CuTe kernel.
# Memory cost: 470MB fp8 → 940MB fp16 (one-time, kept in cache).
# This is intentionally suboptimal vs the sm_100 tcgen05 path; it's enough to
# get H100 compile/correctness validation moving.
# ─────────────────────────────────────────────────────────────────────────────
_W2_F16_CACHE = {}


def _get_w2_f16(weights_fp8: torch.Tensor, scale_fp32: torch.Tensor) -> torch.Tensor:
    key = (weights_fp8.data_ptr(), scale_fp32.data_ptr())
    cached = _W2_F16_CACHE.get(key)
    if cached is not None:
        return cached
    # weights_fp8: [E, H, I], scale: [E, H/128, I/128]. Broadcast scale to
    # full shape and multiply in fp32 for accuracy, cast to fp16.
    QB = 128
    scale_b = scale_fp32.repeat_interleave(QB, dim=1).repeat_interleave(QB, dim=2)
    w_f16 = (weights_fp8.to(torch.float32) * scale_b).to(torch.float16)
    _W2_F16_CACHE[key] = w_f16
    return w_f16

# ─────────────────────────────────────────────────────────────────────────────
# CuTe wgmma TF32 grouped GEMM2 (H100 dev)
# ─────────────────────────────────────────────────────────────────────────────
_USE_CUTE = False
_GEMM2_CUTE = None
_GEMM2_COMPILED = None
_CUTE_IMPORT_ERR = None

# wgmma instruction shape: M=64 fixed, N ∈ {8..256 step 8}, K=8 (TF32)
_BLOCK_M_CUTE = 64
_BLOCK_N_CUTE = 128
_BLOCK_K_CUTE = 128  # matches SB block size

# Conservative upper bound on grid M dim. Worst case: all tokens (incl. dup
# routes) land on a single expert. T_max≈16K * TOP_K=8 = 128K, divided by
# E_LOCAL=32 if balanced is ~4K, but worst case 128K → 128K/64 = 2K tiles.
# We pick 256 as a hard cap — exceeding workloads will under-cover, but the
# real dataset max is ~512 tokens/expert so 256 tiles handles 16K tokens/expert.
_MAX_M_TILES_CUTE = 256

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cute_utils
    import cutlass.utils.hopper_helpers as sm90_utils
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from cutlass.cute.nvgpu import warpgroup, cpasync, CopyUniversalOp

    # wgmma F16 instruction K dim is 16 (TF32 was 8). M=64 fixed, N=128.
    _MMA_INST_MNK = (_BLOCK_M_CUTE, _BLOCK_N_CUTE, 16)
    _CTA_TILE_MNK = (_BLOCK_M_CUTE, _BLOCK_N_CUTE, _BLOCK_K_CUTE)
    _THREADS_PER_CTA = 128  # one warpgroup

    class GEMM2_WGMMA:
        def __init__(self):
            # F16 mantissa = 10 bits (vs BF16 7 bits, vs TF32 10 bits).
            # F16 dynamic range ±65504 covers SwiGLU O(1000+) outputs;
            # mantissa same as TF32 so accuracy should match TF32 path.
            # CHANGELOG L23 rejected BF16, F16 was never tested.
            self.ab_dtype = cutlass.Float16
            self.acc_dtype = cutlass.Float32

        @cute.jit
        def __call__(self, A, B, SA, EO, C, stream):
            # warpgroup.MmaF16BF16Op(ab_dtype, acc_dtype, instruction_shape,
            #                        a_src, a_major_mode, b_major_mode)
            op = warpgroup.MmaF16BF16Op(
                self.ab_dtype, self.acc_dtype,
                _MMA_INST_MNK,
                warpgroup.OperandSource.SMEM,
                warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            )
            tiled_mma = cute.make_tiled_mma(op)

            # sm90 helpers take a LayoutEnum (ROW_MAJOR/COL_MAJOR), not a
            # TiledMma. Both A and B are K-major (last dim contiguous), so
            # ROW_MAJOR for both.
            a_smem_layout = sm90_utils.make_smem_layout_a(
                cute_utils.LayoutEnum.ROW_MAJOR, _CTA_TILE_MNK,
                self.ab_dtype, num_stages=1,
            )
            b_smem_layout = sm90_utils.make_smem_layout_b(
                cute_utils.LayoutEnum.ROW_MAJOR, _CTA_TILE_MNK,
                self.ab_dtype, num_stages=1,
            )

            n_tiles = H // _BLOCK_N_CUTE  # 56
            self._gemm2_kernel(
                tiled_mma, a_smem_layout, b_smem_layout,
                A, B, SA, EO, C,
            ).launch(
                grid=[_MAX_M_TILES_CUTE, n_tiles, E_LOCAL],
                block=[_THREADS_PER_CTA, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def _gemm2_kernel(
            self, tiled_mma, a_smem_layout, b_smem_layout,
            A, B, SA, EO, C,
        ):
            ab_dtype: cutlass.Constexpr = self.ab_dtype
            acc_dtype: cutlass.Constexpr = self.acc_dtype

            tidx, _, _ = cute.arch.thread_idx()
            pid_m, pid_n, pid_e = cute.arch.block_idx()

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
                # sm90 returns a ComposedLayout(swizzle ∘ affine). wgmma's
                # make_fragment_A requires affine-only operand layouts, with
                # the swizzle moved to the pointer. Pass swizzle separately
                # to allocate_tensor so the runtime sets up the descriptor.
                alloc = cute_utils.SmemAllocator()
                sA = alloc.allocate_tensor(
                    element_type=ab_dtype,
                    layout=a_smem_layout.outer,
                    byte_alignment=128,
                    swizzle=a_smem_layout.inner,
                )
                sB = alloc.allocate_tensor(
                    element_type=ab_dtype,
                    layout=b_smem_layout.outer,
                    byte_alignment=128,
                    swizzle=b_smem_layout.inner,
                )

                # ── Partition sA/sB into MMA mode profile, then build frags ──
                # wgmma make_fragment_A expects (MMA, MMA_M, MMA_K, ...) shape.
                # tiled_mma.get_slice(0).partition_{A,B}(s{A,B}) does the
                # partition. sm100 tcgen05 hides this inside make_fragment;
                # sm90 wgmma we partition manually.
                thr_mma = tiled_mma.get_slice(0)
                tCsA = thr_mma.partition_A(sA)
                tCsB = thr_mma.partition_B(sB)
                tCrA = tiled_mma.make_fragment_A(tCsA)
                tCrB = tiled_mma.make_fragment_B(tCsB)

                # ── Register accumulator (no TMEM on Hopper) ──────────────────
                acc_shape = tiled_mma.partition_shape_C(_CTA_TILE_MNK[:2])
                acc_frag = tiled_mma.make_fragment_C(acc_shape)
                acc_frag.fill(acc_dtype(0.0))

                num_k_blocks = cute.size(tCrA, mode=[2])

                K_DIM: cutlass.Constexpr = I
                num_k_tiles: cutlass.Constexpr = K_DIM // _BLOCK_K_CUTE  # 16

                # Build identity-layout (non-swizzled) views over sA/sB so we
                # can scalar-store from threads. The swizzle stays on the smem
                # *allocation* (so wgmma reads it correctly), but stores via
                # the identity view bypass the swizzle's runtime coord check.
                # Equivalent to writing through ptr+offset with stride (BK,1)
                # / (BK,1).
                sA_flat = cute.make_tensor(
                    sA.iterator,
                    cute.make_layout(
                        (_BLOCK_M_CUTE, _BLOCK_K_CUTE),
                        stride=(_BLOCK_K_CUTE, 1),
                    ),
                )
                sB_flat = cute.make_tensor(
                    sB.iterator,
                    cute.make_layout(
                        (_BLOCK_N_CUTE, _BLOCK_K_CUTE),
                        stride=(_BLOCK_K_CUTE, 1),
                    ),
                )

                for kb in cutlass.range_constexpr(num_k_tiles):
                    k_start = kb * cutlass.Int32(_BLOCK_K_CUTE)

                    # ── Load A tile (fp16, pre-scaled by SA) → smem ───────────
                    # A came in as f16 already divided by per-row SA scale, so
                    # this is a plain f16→f16 copy. SA is applied in epilogue.
                    ELEMS_PER_THREAD_A: cutlass.Constexpr = (
                        _BLOCK_M_CUTE * _BLOCK_K_CUTE // _THREADS_PER_CTA
                    )
                    for i in cutlass.range_constexpr(ELEMS_PER_THREAD_A):
                        flat_idx = tidx * cutlass.Int32(ELEMS_PER_THREAD_A) + cutlass.Int32(i)
                        a_m = flat_idx // cutlass.Int32(_BLOCK_K_CUTE)
                        a_k = flat_idx - a_m * cutlass.Int32(_BLOCK_K_CUTE)
                        g_m = e_start + m_local_start + a_m
                        g_k = k_start + a_k
                        sA_flat[a_m, a_k] = A[g_m, g_k]

                    # ── Load B tile (fp16, pre-dequanted in wrapper) → smem ───
                    # Path B sidesteps the sm_90 fp8 vector-cast restriction
                    # by passing a fp16 B already multiplied by SB scale at
                    # the Python wrapper layer. Kernel just copies fp16→fp16.
                    n_start = pid_n * cutlass.Int32(_BLOCK_N_CUTE)
                    ELEMS_PER_THREAD_B: cutlass.Constexpr = (
                        _BLOCK_N_CUTE * _BLOCK_K_CUTE // _THREADS_PER_CTA
                    )
                    for i in cutlass.range_constexpr(ELEMS_PER_THREAD_B):
                        flat_idx = tidx * cutlass.Int32(ELEMS_PER_THREAD_B) + cutlass.Int32(i)
                        b_n = flat_idx // cutlass.Int32(_BLOCK_K_CUTE)
                        b_k = flat_idx - b_n * cutlass.Int32(_BLOCK_K_CUTE)
                        sB_flat[b_n, b_k] = B[pid_e, n_start + b_n, k_start + b_k]

                    cute.arch.sync_threads()
                    warpgroup.fence()

                    # ── wgmma F16 MMA (whole warpgroup cooperative) ───────────
                    # tCrA layout has 4 modes: (MMA, MMA_M, MMA_K, stages).
                    # Slice with full 4-tuple, keep MMA/MMA_M/stages, pick MMA_K.
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, None)
                        cute.gemm(
                            tiled_mma, acc_frag,
                            tCrA[k_block_coord], tCrB[k_block_coord], acc_frag,
                        )

                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                    cute.arch.sync_threads()

                # ── Epilogue: register fragment → gmem ────────────────────────
                # Build a gmem tile view of C[e_start+m_local_start:..,
                # pid_n*BLOCK_N:..] with shape (BLOCK_M, BLOCK_N), then
                # partition_C using the same TiledMma slice and cute.copy
                # from acc_frag.
                row_base = e_start + m_local_start
                col_base = pid_n * cutlass.Int32(_BLOCK_N_CUTE)
                gC_ptr = cute.make_ptr(
                    acc_dtype,
                    (C.iterator + row_base * cutlass.Int32(H) + col_base).toint(),
                    mem_space=cute.AddressSpace.gmem, assumed_align=4,
                )
                gC_tile = cute.make_tensor(
                    gC_ptr,
                    cute.make_layout(
                        (_BLOCK_M_CUTE, _BLOCK_N_CUTE),
                        stride=(H, 1),
                    ),
                )
                tCgC = thr_mma.partition_C(gC_tile)
                # cute.copy(atom, src, dst) for register→gmem also produced
                # silent no-store. Fall back to element-wise scalar store —
                # identical pattern to sA_flat[a_m, a_k] = ... in the K-loop,
                # which we know writes correctly.
                num_acc_elems: cutlass.Constexpr = cute.size(acc_frag)
                for i in cutlass.range_constexpr(num_acc_elems):
                    tCgC[i] = acc_frag[i]
                # NOTE: SA per-row scale is applied in the Python wrapper
                # after this kernel returns (gemm2_out *= sa.unsqueeze(1)).
                _ = SA  # keep reference to silence unused-arg lint

    def _compile_gemm2(instance):
        T_ = cute.sym_int()
        # A is now fp16 (pre-scaled by per-row SA in wrapper).
        A_fake = make_fake_compact_tensor(
            dtype=cutlass.Float16, shape=(T_, I), stride_order=(1, 0),
            assumed_align=2,
        )
        # B is fp16 (pre-dequanted in wrapper), SB is gone.
        B_fake = make_fake_compact_tensor(
            dtype=cutlass.Float16, shape=(E_LOCAL, H, I),
            stride_order=(2, 1, 0), assumed_align=16,
        )
        # SA: per-row fp32 scale [total_tokens] — applied in epilogue.
        SA_fake = make_fake_compact_tensor(
            dtype=cutlass.Float32, shape=(T_,), stride_order=(0,),
            assumed_align=4,
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
            A_fake, B_fake, SA_fake, EO_fake, C_fake, stream,
            options="--enable-tvm-ffi",
        )

    # NO fallback — let cute.compile failures propagate so they show up in
    # the evaluator's per-workload log (printed by run_modal.print_results
    # via worst-failure dump).
    _GEMM2_CUTE = GEMM2_WGMMA()
    _GEMM2_COMPILED = _compile_gemm2(_GEMM2_CUTE)
    _USE_CUTE = True
    print("[kernel_cute_h100] wgmma TF32 GEMM2 compiled OK", flush=True)

except Exception:
    # Same: re-raise. Module-load failure must be visible.
    print("[kernel_cute_h100] CuTe setup failed — see traceback below", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    raise

# ─────────────────────────────────────────────────────────────────────────────
# FIB_PROFILE: per-stage CUDA event timing
# ─────────────────────────────────────────────────────────────────────────────
_PROFILE = os.environ.get("FIB_PROFILE", "0") == "1"


class _StageTimer:
    def __init__(self):
        self.events = []
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
        print(f"[kernel_cute_h100 stages] {line}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — same signature as v11/kernel_cute.py
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def kernel(
    routing_logits, routing_bias,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
    output,
):
    T = hidden_states.shape[0]
    device = hidden_states.device
    local_start = int(local_expert_offset)
    timer = _StageTimer()

    timer.start("routing")
    topk_idx = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
    topk_weights = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
    _v11._routing_kernel[(T,)](
        routing_logits, routing_bias, topk_idx, topk_weights,
        T, routed_scaling_factor,
    )
    timer.stop()

    timer.start("count")
    expert_counts = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    _v11._count_expert_tokens[(T,)](topk_idx, expert_counts, local_start, T)
    timer.stop()

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

    timer.start("gather")
    sorted_A = hidden_states[sorted_token_ids].contiguous()
    sorted_A_scale = hidden_states_scale[:, sorted_token_ids].contiguous()
    timer.stop()

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

    timer.start("swiglu")
    # Path B (sm_90): SwiGLU + per-row fp16 scale. A enters GEMM2 as fp16
    # already divided by SA so values stay in [-32000, 32000]; SA is multiplied
    # back into gemm2_out after GEMM2.
    swiglu_fp16 = torch.empty((total_tokens, I), dtype=torch.float16, device=device)
    sa_per_row = torch.empty(total_tokens, dtype=torch.float32, device=device)
    _swiglu_to_fp16_perrow[(triton.cdiv(total_tokens, 32),)](
        gemm1_out, swiglu_fp16, sa_per_row,
        total_tokens,
        I=I, BLOCK_M=32, BLOCK_I=128,
    )
    timer.stop()

    # zero, not empty — the CuTe kernel only writes rows that fall within
    # M_e per expert; out-of-range rows must be 0 so the subsequent SA
    # multiply doesn't propagate garbage / NaN.
    gemm2_out = torch.zeros((total_tokens, H), dtype=torch.float32, device=device)

    timer.start("gemm2")
    # Path B (sm_90): pre-dequant gemm2_weights to fp16 once per weight tensor
    # (cached). The CuTe kernel sidesteps the sm_90 fp8 vector-cast restriction
    # by reading already-dequanted fp16 B. Grid M dim is fixed at
    # _MAX_M_TILES_CUTE; out-of-range CTAs early-out via the per-expert
    # `m_local_start < M_e` check.
    gemm2_weights_f16 = _get_w2_f16(gemm2_weights, gemm2_weights_scale)
    _GEMM2_COMPILED(
        swiglu_fp16, gemm2_weights_f16, sa_per_row,
        expert_offsets_gpu, gemm2_out,
    )
    # Apply per-row SA scale that was divided out of A pre-GEMM2.
    gemm2_out *= sa_per_row.unsqueeze(1)
    # Diagnostic: print stats on first call to track down the NaN source.
    if not hasattr(kernel, "_h100_dbg_done"):
        torch.cuda.synchronize()
        print(f"[h100_dbg] T={total_tokens}", flush=True)
        print(f"[h100_dbg] sa_per_row min={sa_per_row.min().item():.3e} "
              f"max={sa_per_row.max().item():.3e} "
              f"has_zero={(sa_per_row == 0).any().item()} "
              f"has_nan={torch.isnan(sa_per_row).any().item()}", flush=True)
        print(f"[h100_dbg] swiglu_fp16 has_inf={torch.isinf(swiglu_fp16).any().item()} "
              f"has_nan={torch.isnan(swiglu_fp16).any().item()} "
              f"absmax={swiglu_fp16.abs().max().item():.3e}", flush=True)
        print(f"[h100_dbg] gemm2_out has_inf={torch.isinf(gemm2_out).any().item()} "
              f"has_nan={torch.isnan(gemm2_out).any().item()} "
              f"absmax={gemm2_out.abs().max().item():.3e}", flush=True)
        # Sample a few rows
        print(f"[h100_dbg] gemm2_out[0, :8]={gemm2_out[0, :8].tolist()}", flush=True)
        print(f"[h100_dbg] gemm2_out[total_tokens//2, :8]="
              f"{gemm2_out[total_tokens // 2, :8].tolist()}", flush=True)
        print(f"[h100_dbg] gemm2_out[-1, :8]={gemm2_out[-1, :8].tolist()}", flush=True)
        kernel._h100_dbg_done = True
    timer.stop()

    timer.start("scatter_out")
    weighted = gemm2_out * sorted_weights.unsqueeze(1)
    out_f32 = torch.zeros((T, H), dtype=torch.float32, device=device)
    out_f32.index_add_(0, sorted_token_ids.long(), weighted)
    output.copy_(out_f32)
    timer.stop()

    timer.report()
