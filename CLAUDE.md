# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a submission for the **MLSys 2026 FlashInfer AI Kernel Generation Contest** targeting the **fused_moe** track. The goal is to write the fastest possible GPU kernel for the DeepSeek-V3 style Fused Mixture-of-Experts operation on NVIDIA B200 (Blackwell, sm_100a) GPUs.

**Kernel definition:** `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- H=7168 (hidden dim), I=2048 (intermediate dim), E_GLOBAL=256 experts, E_LOCAL=32 per node, TOP_K=8
- FP8 (e4m3fn) weights with block-scale quantization (128-element blocks)
- Destination passing style (DPS): output tensor is the last argument

## Setup

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
# Or install flashinfer-bench from source for latest changes:
git clone https://github.com/flashinfer-ai/flashinfer-bench.git && cd flashinfer-bench && pip install -v -e .
```

Download the dataset (one-time):
```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
export FIB_DATASET_PATH=/path/to/mlsys26-contest
```

## Key Commands

```bash
# Pack source files into solution.json (required before local/modal runs)
python scripts/pack_solution.py

# Run benchmark locally (requires FIB_DATASET_PATH env var and local CUDA GPU)
python scripts/run_local.py

# One-time Modal cloud setup
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/mlsys26-contest

# Run benchmark on B200 via Modal
modal run scripts/run_modal.py
```

## Submission

```bash
python scripts/pack_solution.py   # generates solution.json
git add solution.json
git commit -m "..."
git tag submission-vX             # pipeline picks up the latest tag per definition
git push && git push --tags
```

## Architecture

**`config.toml`** — controls what gets packed:
- `solution.definition`: the kernel ID being targeted
- `build.language`: `triton` or `cuda`
- `build.entry_point`: `kernel.py::kernel` (for Triton)
- `build.destination_passing_style`: `true` (output is the last arg)

**`solution/triton/kernel.py`** — the active Triton implementation. The evaluation pipeline reads source from `solution/triton/` when `language = "triton"`.

**`solution/triton/CHANGELOG.md`** — tracks the version history with performance results. Update this whenever the kernel changes.

**`scripts/pack_solution.py`** — reads `config.toml`, collects all files from the language-specific source dir, and serializes them into `solution.json`. The evaluation pipeline can also pack directly from the repo if `solution.json` doesn't exist.

## Kernel Architecture (current: v3 in kernel.py)

The kernel has three stages:

1. **Routing** (`_compute_routing`): Pure PyTorch. Sigmoid + bias → group-wise top-2 → top-4 groups → global top-8 experts. Returns compact `[T, 8]` expert indices and weights.

2. **Per-expert GEMM loop**: Iterates over 32 local experts in Python. For each expert with active tokens:
   - `_gemm1`: `dequant(FP8 hidden) @ dequant(FP8 W1).T → [Tk, 2*I]` float32
   - SwiGLU: `silu(x2) * x1` (PyTorch)
   - `_gemm2`: `float32_intermediate @ dequant(FP8 W2).T → [Tk, H]` float32
   - Weighted accumulation into output via `index_add_`

3. **GEMM kernels** (`_fp8_fp8_gemm`, `_f32_fp8_gemm`): Triton kernels with `@triton.autotune` over BLOCK_M ∈ {16, 32, 64, 128}, num_warps ∈ {4, 8}, num_stages ∈ {3, 4}. BLOCK_N=128 and BLOCK_K=128 are fixed to align exactly with FP8 block-scale boundaries.

## Evaluation Environment

Official evaluation runs on **bare-metal B200 (sm_100a)** with locked clocks — not Modal. Modal (sm100) is for development only.

| | Value |
|---|---|
| CUDA | 13.2 |
| PyTorch | 2.12.0+cu132 |
| Triton | 3.6.0 |

**Correctness tolerances (MoE track):** atol=1, rtol=0.3, required_matched_ratio=0.9. A workload fails only if both abs_error > atol AND rel_error > rtol.

**Scoring:** `speedup = FlashInfer_baseline_latency / your_latency`. Arithmetic mean over all workloads. Any failing workload zeros the entire definition's score.

## Optimization Priorities

The main bottleneck is the **sequential Python expert loop** (32 iterations, each launching separate Triton kernels). High-impact improvements in order:

1. **Batched/grouped GEMM across all experts** — eliminate the Python for-loop; process all expert–token pairs in one kernel launch
2. **Fused routing Triton kernel** — replace PyTorch routing ops with a single GPU kernel (planned as v4 per CHANGELOG)
3. **SwiGLU fused into GEMM1 epilogue** — avoids writing the `[Tk, 4096]` intermediate to HBM (v2 attempted this but regressed; revisit with better tile sizing)
4. **FP8 quantization of GEMM2 input** — GEMM2 currently takes float32 input; quantizing to FP8 halves memory bandwidth

## Known Constraints

- **No external library calls at runtime**: flashinfer, deep_gemm, etc. cannot be called — all code must be in the submission sources. CUTLASS/CuTe headers and cuBLAS are permitted.
- **No variadic kernel signatures**: will fail builder validation.
- **No pre-built cubins**: kernels must be compiled from source during evaluation.
- **No network access** in the evaluation environment.
- `solution_dir` is not supported in config.toml — use `source_dir` instead.
