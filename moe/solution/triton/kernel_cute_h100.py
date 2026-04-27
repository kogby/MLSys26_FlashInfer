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

H = _v11.H
I = _v11.I
E_LOCAL = _v11.E_LOCAL
TOP_K = _v11.TOP_K

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

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cute_utils
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from cutlass.cute.nvgpu import warpgroup, cpasync

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
        def __call__(self, A, B, SB, EO, C, max_M_tiles, stream):
            # warpgroup.MmaF16BF16Op(ab_dtype, acc_dtype, instruction_shape,
            #                        a_src, a_major_mode, b_major_mode)
            op = warpgroup.MmaF16BF16Op(
                self.ab_dtype, self.acc_dtype,
                _MMA_INST_MNK,
                warpgroup.OperandSource.SMEM,
                warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            )
            tiled_mma = cute.make_tiled_mma(op)

            # warpgroup smem layout helpers (vs sm100_utils for tcgen05)
            a_smem_atom = warpgroup.make_smem_layout_atom(
                warpgroup.SmemLayoutAtomKind.MN_INTER, self.ab_dtype,
            )
            b_smem_atom = warpgroup.make_smem_layout_atom(
                warpgroup.SmemLayoutAtomKind.MN_INTER, self.ab_dtype,
            )
            a_smem_layout = cute.tile_to_shape(
                a_smem_atom, (_BLOCK_M_CUTE, _BLOCK_K_CUTE), order=(0, 1),
            )
            b_smem_layout = cute.tile_to_shape(
                b_smem_atom, (_BLOCK_N_CUTE, _BLOCK_K_CUTE), order=(0, 1),
            )

            n_tiles = H // _BLOCK_N_CUTE  # 56
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

            # ── per-expert M range ────────────────────────────────────────────
            e_start = EO[pid_e]
            e_end = EO[pid_e + 1]
            M_e = e_end - e_start
            m_local_start = pid_m * cutlass.Int32(_BLOCK_M_CUTE)
            if m_local_start >= M_e:
                return

            # ── SMEM allocation ───────────────────────────────────────────────
            alloc = cute_utils.SmemAllocator()
            sA = alloc.allocate_tensor(
                element_type=ab_dtype, layout=a_smem_layout, byte_alignment=128,
            )
            sB = alloc.allocate_tensor(
                element_type=ab_dtype, layout=b_smem_layout, byte_alignment=128,
            )

            # ── Register accumulator (no TMEM on Hopper) ──────────────────────
            acc_shape = tiled_mma.partition_shape_C(_CTA_TILE_MNK[:2])
            acc_frag = tiled_mma.make_fragment_C(acc_shape)
            acc_frag.fill(acc_dtype(0.0))

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            num_k_blocks = cute.size(tCrA, mode=[2])

            K_DIM: cutlass.Constexpr = I
            num_k_tiles: cutlass.Constexpr = K_DIM // _BLOCK_K_CUTE  # 16

            for kb in cutlass.range_constexpr(num_k_tiles):
                k_start = kb * cutlass.Int32(_BLOCK_K_CUTE)

                # ── Load A tile (fp32) → cast TF32 → smem ─────────────────────
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

                # ── Load B tile (fp8) → dequant + scale → cast TF32 → smem ────
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
                warpgroup.fence()

                # ── wgmma TF32 MMA (whole warpgroup cooperative) ──────────────
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx)
                    cute.gemm(
                        tiled_mma, acc_frag,
                        tCrA[k_block_coord], tCrB[k_block_coord], acc_frag,
                    )

                warpgroup.commit_group()
                warpgroup.wait_group(0)
                cute.arch.sync_threads()

            # ── Epilogue: register fragment → gmem with M_e mask ──────────────
            # acc_frag is per-thread; partition_C tells us which output rows
            # this thread owns. For wgmma+M=64+128 threads, each thread writes
            # 32 output elements (split across rows/cols by tiled_mma layout).
            thr_acc = tiled_mma.get_slice(tidx).partition_C(
                cute.make_tensor(
                    cute.make_ptr(
                        acc_dtype, 0, mem_space=cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    cute.make_layout(_CTA_TILE_MNK[:2], stride=(_BLOCK_N_CUTE, 1)),
                )
            )
            # Iterate the per-thread frag and write to C[g_m, g_n]
            for i in cutlass.range_constexpr(cute.size(acc_frag)):
                # Get logical (m,n) for this thread's i-th register element
                m_local, n_local = thr_acc.crd2idx_inv(i)  # may not be the API
                m_global = e_start + m_local_start + m_local
                n_global = pid_n * cutlass.Int32(_BLOCK_N_CUTE) + n_local
                if (m_local_start + m_local) < M_e:
                    C[m_global, n_global] = acc_frag[i]

    def _compile_gemm2(instance):
        T_ = cute.sym_int()
        max_M_tiles_ = cute.sym_int()
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
    _v11._swiglu_inplace[
        (triton.cdiv(total_tokens, 32), triton.cdiv(I, 128))
    ](gemm1_out, total_tokens, I=I, BLOCK_M=32, BLOCK_I=128)
    timer.stop()

    assert gemm1_out.stride() == (2 * I, 1), (
        f"Unexpected gemm1_out stride: {gemm1_out.stride()}, expected ({2*I}, 1)"
    )
    gemm2_out = torch.empty((total_tokens, H), dtype=torch.float32, device=device)

    timer.start("gemm2")
    # No fallback: if _USE_CUTE is False, module-load already raised.
    max_M_e = max(token_counts_cpu) if token_counts_cpu else 0
    max_M_tiles = max((max_M_e + _BLOCK_M_CUTE - 1) // _BLOCK_M_CUTE, 1)
    _GEMM2_COMPILED(
        gemm1_out, gemm2_weights, gemm2_weights_scale,
        expert_offsets_gpu, gemm2_out,
        max_M_tiles, torch.cuda.current_stream(),
    )
    timer.stop()

    timer.start("scatter_out")
    weighted = gemm2_out * sorted_weights.unsqueeze(1)
    out_f32 = torch.zeros((T, H), dtype=torch.float32, device=device)
    out_f32.index_add_(0, sorted_token_ids.long(), weighted)
    output.copy_(out_f32)
    timer.stop()

    timer.report()
