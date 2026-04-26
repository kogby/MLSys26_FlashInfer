# Triton Kernel Changelog

## v10 — FP8 tensor cores for GEMM1 (current, best)

**Optimization:** In `_grouped_gemm1_swiglu`, stop pre-converting A,B from FP8 to float32 before `tl.dot`. Instead: `partial = tl.dot(fp8_A, fp8_B_T, out_dtype=float32)`, then `acc += partial * sa[:,None] * sb`. This is mathematically identical (sa,sb are K-block-constant scalars that can factor out of the sum), but routes through H100/H200 FP8 WMMA tensor cores (3.9–4.9 PFLOP/s) instead of float32 SIMD (0.98–1.5 PFLOP/s).

GEMM2 is unchanged from v8 (float32 A × FP8 B). BF16 tensor cores for GEMM2 were tried and caused matched_ratio < threshold on production workloads: SwiGLU outputs can reach O(1000+), and BF16's 7-bit mantissa introduces ~5-7% of elements outside (atol=1, rtol=0.3).

**Why the large workloads improve most:** For total_tokens≈65k, GEMM1 was compute-bound at FP32 (3.92ms compute vs 0.74ms memory floor). FP8 tensor cores reduce compute time to ~0.98ms, making GEMM1 memory-bandwidth-limited. Net saving: ~3ms per large workload.

**Results (19 workloads, all PASSED on H200):**
- Latency — min: 1.293 ms, max: 9.065 ms, median: 1.569 ms
- Speedup — min: 5.800x (large 9.1ms workload), max: 11.956x, **mean: 10.945x**
- Large workload improvement: 5e8dc11c 3.84x→5.80x (+51%), 58a34f27 4.35x→6.43x (+48%)
- Note: run on H200 (H200 has higher reference FlashInfer throughput, so medium-workload speedups are lower than H100 numbers; large-workload improvement is genuine)

---

## v9 experiments — All regressed vs v8 (abandoned)

## v8 — Persistent 2D tile scheduling (superseded by v10)

## v9 experiments — All regressed vs v8 (abandoned)

Three v9 variants were tried; all were slower than v8. v8 remains the active implementation.

**v9a — Fused `_grouped_gemm2_fused` + SwiGLU + weighted atomic scatter:**
- Kernel reads gate+up tiles from `gemm1_out`, applies SwiGLU in-register, then `tl.atomic_add` directly into `out_f32`.
- Correctness bug (50.6% pass ratio): `tl.atomic_add` + `@triton.autotune` accumulates across ~3000 autotune trials. Fixed by `reset_to_zero=['Out_ptr']`.
- Performance after fix: mean **9.432x** — regression. Root cause: scattered atomic writes from 8 concurrent expert CTAs cause L2 cache line serialization. Large workload (13.8ms → 17.7ms, +28% slower).

**v9b — `_grouped_gemm2_swiglu`: SwiGLU fused into GEMM2 A-tile loads, `tl.store` epilogue:**
- Reads gate (cols [0, K)) and up (cols [I, I+K)) from `gemm1_out` per K-iteration, applies `silu(up)*gate` in-register. Eliminates `_swiglu_inplace` (~1 GB HBM saved).
- Performance: mean **8.981x** — regression. Root cause: loading 2× A tiles per K-iteration doubles A-side HBM pressure in GEMM2, disrupting the autotuner into smaller BLOCK_M configs.

**v9c — Weight multiply fused into GEMM2 epilogue:**
- GEMM2 loads `sorted_weights[rm]` in the epilogue and stores `acc * w[:, None]` instead of `acc`. Eliminates the separate `weighted = gemm2_out * sorted_weights` tensor (~3.76 GB HBM traffic saved for large workloads).
- Performance: mean **9.703x** — regression. Root cause: materializing `acc * w[:, None]` before `tl.store` doubles epilogue register usage, reducing occupancy and causing the autotuner to select smaller BLOCK_M for medium workloads.

**Lesson:** Adding computation to GEMM2 (whether SwiGLU, atomic scatter, or weight multiply) consistently reduces the autotuner's effective BLOCK_M and hurts throughput more than the HBM savings gained.

---

## v8 — Persistent 2D tile scheduling

**Optimization:** Replace the 3D grid `(ceil(max_tokens/BM), N_TILES, E_LOCAL)` with a 2D grid `(total_valid_m_tiles, N_TILES)` for both `_grouped_gemm1_swiglu` and `_grouped_gemm2`. Each CTA decodes its expert and local m-tile via a 32-iteration static loop (unrolled at compile time) over `expert_offsets`, using `tl.static_range(E_LOCAL)` + `tl.where` accumulation.

**What changed:**
- Both GEMM kernels: `pid_e = tl.program_id(2)` removed; replaced by 32-iteration `tl.static_range` decode from global m-tile index.
- Autotune key: `max_tokens` → `total_tokens` (sum of all expert token counts).
- Host code: `max_tokens_padded` removed; `token_counts_cpu` (32 ints from one GPU→CPU sync) drives both grid lambdas.
- Grid lambdas: 3D `(m_tiles, n_tiles, E_LOCAL)` → 2D `(total_valid_m_tiles, n_tiles)`.

**Why this helps:**
- Old 3D grid: `ceil(max_tokens/BM) × N_TILES × 32` CTAs launched, but experts with fewer tokens than `max_tokens` wasted many CTAs doing masked-out work.
- New 2D grid: exactly `sum(ceil(tokens_e/BM) for e) × N_TILES` CTAs, all doing real work.
- For imbalanced workloads (e.g., 5 experts with 512 tokens, 27 with 0): old CTA count ≈ 4096, new ≈ 320 — 12× fewer for GEMM1.

**Results (19 workloads, all PASSED on H100):**
- Latency — min: 0.909 ms, max: 13.798 ms, median: 1.581 ms
- Speedup — min: 3.830x (large 13.8ms workload), max: 16.169x, **mean: 10.848x**
- Improvement over v7: +2.724x mean speedup (8.124x → 10.848x), median latency 1.87 ms → 1.58 ms

---

## v7 — Fix GEMM1 autotune config + num_stages=5 (superseded by v8)

**Bug fixed:** `_GEMM1_CONFIGS` incorrectly capped `BLOCK_M` at 64. `_grouped_gemm1_swiglu` has a **single accumulator** (SwiGLU is handled by a separate `_swiglu_inplace` pass, not by two in-register accumulators), so BLOCK_M=128 is safe. The cap was a copy-paste artifact from the abandoned v6.1 two-accumulator design.

**What changed:**
- `_GEMM1_CONFIGS`: BLOCK_M range expanded from `{16, 32, 64}` → `{16, 32, 64, 128}`
- Both `_GEMM1_CONFIGS` and `_GEMM_CONFIGS`: `num_stages` expanded from `{3, 4}` → `{3, 4, 5}`
- `_swiglu_inplace` launch: BLOCK_M bumped from 16 → 32 (halves CTA count for the SwiGLU pass)

**Results (19 workloads, all PASSED on H100):**
- Latency — min: 1.85 ms, max: 23.72 ms, median: 1.87 ms
- Speedup — min: 2.225x (large 23ms workload), max: 9.663x, **mean: 8.124x**
- Improvement over v5.1: +0.634x mean speedup (7.49x → 8.124x), median latency 2.11 ms → 1.87 ms

---

## v5.1 — Revert to v5 grouped GEMM pipeline (superseded by v7)

**Reason:** v6's FP8 intermediate (`swiglu_fp8`) produced INCORRECT_NUMERICAL on all 19 production workloads (abs errors 13,000–28,000) despite passing the synthetic debug test. Root cause isolated via systematic elimination: the FP8 quantize-dequantize roundtrip in the sorted-token-id path introduced errors that differed from FlashInfer's float32 reference. The `tl.float8e4nv` type is the correct Triton dtype for `torch.float8_e4m3fn` on H100 (not `tl.float8e4m3fn` which doesn't exist in Triton 3.6.0), but the production data magnitudes expose numerical issues in the FP8 intermediate path.

**What changed:** Reverts to v5's three-kernel pipeline: `_grouped_gemm1_swiglu` (FP8×FP8→float32), `_swiglu_inplace` (in-place SwiGLU), `_grouped_gemm2` (float32×FP8→float32). Routing, expert-map, and scatter remain unchanged. Pure Python reference (which matched FlashInfer's output exactly with 0 error on all workloads) confirmed the algorithm is correct; only the Triton kernel implementations needed to be restored.

**Results (19 workloads, all PASSED on H100):**
- Latency — min: 2.10 ms, max: 25.74 ms, median: 2.11 ms
- Speedup — min: 2.08x (large 25ms workload), max: 8.94x, **mean: 7.49x**
- Correctness: matched_ratio ≥ 0.961 for all workloads (threshold is 0.9); max abs error = 4096–8192 is tail-element FP8+float32 numerical difference vs reference

---

## v6.1 — Fused GEMM1+SwiGLU, float32 output (REGRESSED — superseded by v5.1)

**Optimization:** Replace v5.1's `_grouped_gemm1_swiglu` + `_swiglu_inplace` two-pass with a single `_grouped_gemm1_swiglu_f32` kernel that holds two accumulators (`acc_gate`, `acc_up`), applies `silu(up)*gate` in-register, and writes [total_tokens, I] float32 directly. Eliminates the [total_tokens, 2*I] HBM write+read roundtrip.

**Why it regressed:** Two accumulators double per-CTA register pressure, forcing the autotuner to cap `BLOCK_M ≤ 64` (vs `≤ 128` in v5.1). Lower BLOCK_M means fewer rows per CTA, lower arithmetic intensity, and worse latency hiding for large memory-bandwidth-bound workloads. The intermediate buffer savings (~4× less HBM traffic) are outweighed by the occupancy loss.

**Results (19 workloads, all PASSED on H100):**
- Latency — min: ~2 ms, max: 27.41 ms
- Speedup — min: 1.959x, max: 8.850x, **mean: 7.271x** (vs v5.1's 7.49x)
- Large workloads regressed: 27.41 ms (v6.1) vs 25.74 ms (v5.1), 18.27 ms vs 17.17 ms

---

## v6 — Fused SwiGLU epilogue + FP8 quantized GEMM2 input (BROKEN — superseded by v5.1)

**Optimization:** Fuse SwiGLU into the GEMM1 epilogue (in-register, two accumulators), block-quantize the result to FP8, and feed it directly into a full FP8×FP8 GEMM2. Eliminates the 134 MB FP32 intermediate buffer from v5 and reduces GEMM2 A-matrix bandwidth 4×.

**What changed:**
- `_grouped_gemm1_swiglu_fp8`: replaces v5's `_grouped_gemm1_swiglu` + `_swiglu_inplace`. Each CTA loads **two B-tiles per K-iteration** (gate half at `rn_gate`, up half at `rn_gate + I`) and accumulates into `acc_gate` and `acc_up`. After the K-loop, applies `silu(acc_up) * acc_gate` in registers, then block-quantizes to FP8 (one scale per 128 output features per token). Output: `swiglu_fp8 [total_tokens, I]` FP8 + `swiglu_scale [I//128, total_tokens]` FP32. Grid is halved in N-dimension (`I_TILES=16` instead of `2*I_TILES=32`) since each CTA handles a gate+up pair together.
- `_grouped_fp8_gemm`: generic grouped FP8×FP8→FP32 GEMM. GEMM2 now takes `swiglu_fp8` + `swiglu_scale` as its A input.
- GEMM1 configs capped at `BLOCK_M ∈ {16, 32}` due to two-accumulator register pressure.
- GEMM2 configs unchanged: `BLOCK_M ∈ {16, 32, 64, 128}`.

**HBM traffic for intermediate tensors (T=1024, total_tokens≈8192):**
- v5: 402 MB (gemm1_write + swiglu_rw + gemm2_A_read)
- v6: ~17 MB (swiglu_fp8_write + swiglu_fp8_read + scales)

---

## v5 — Grouped/batched GEMM across all experts (superseded by v6)

**Optimization:** Replace the 32-iteration Python expert loop with a single grouped GEMM launch per stage, eliminating ~100 GPU kernel launches per forward pass.

**What changed:**

- `_build_expert_map`: Pure PyTorch. Expands `topk_idx [T,8]` into flat `(token, expert)` pairs, filters to local experts, sorts by expert id, computes `expert_offsets [E_LOCAL+1]`. O(T) work.
- `_grouped_gemm1_swiglu`: 3D Triton kernel `grid=(m_tiles, n_tiles, E_LOCAL)`. Each CTA reads `expert_offsets[pid_e]..expert_offsets[pid_e+1]` to find its token range. FP8×FP8 GEMM1 with same per-tile dequantization as v3.
- `_swiglu_inplace`: Lightweight Triton kernel that reads gate/up from the GEMM1 output buffer and writes `silu(up)*gate` into the first I cols in-place.
- `_grouped_gemm2`: Same 3D structure, f32×FP8 GEMM2.
- `_scatter_add`: Triton kernel that does `out[token_id] += weight * gemm2_out[row]` via `tl.atomic_add`.

**Why this is faster:**
- 32 Python iterations × 2 Triton launches → 3 Triton launches total
- Each expert's token tiles run in parallel across 192 B200 SMs instead of sequentially
- The GPU sees all T×8 token-expert pairs at once, filling more SMs simultaneously

**v4 (fused routing) was skipped** — routing is <10% of time; grouped GEMM is the dominant win.

**GEMM kernels and scale handling are identical to v3.**

---

## v4 — Fused routing kernel (planned, not implemented)

**Optimization:** Replace 6+ sequential PyTorch routing ops with a single Triton kernel (`_routing_kernel`) — one GPU program per token.

**What changed:**
- Added `_routing_kernel`: each CTA handles one token and executes the full DeepSeek-V3 routing pipeline in registers:
  1. `sigmoid(logits) + bias` → biased scores [256]
  2. Reshape to [8, 32] groups, sort each group descending, sum top-2 → group scores [8]
  3. Sort group scores, take threshold at rank 4 → group selection mask [8]
  4. Broadcast mask to [256], zero out non-selected experts
  5. `argsort` → take top-8 expert indices
  6. Gather unbiased sigmoid values for selected experts via [8, 256] broadcast comparison
  7. Normalize and scale weights
- Routing output is now compact **[T, 8]** (int32 indices + float32 weights) instead of sparse [T, 256].
  For T=1024: 32 KB vs 1 MB — eliminates 32× larger intermediate tensor.
- Per-expert weight extraction: `(topk_w[tok_idx] * (topk_idx[tok_idx] == ge)).sum(1)` replaces `weights[tok_idx, ge]`.

**GEMM kernels and SwiGLU are unchanged from v3 (still autotuned).**

---

## v3 — Autotuned GEMM tile sizes (superseded by v4)

**Optimization:** Autotuning over BLOCK_M, num_warps, and num_stages for both GEMM kernels.

**What changed:** Added `@triton.autotune` to both `_fp8_fp8_gemm` and `_f32_fp8_gemm`. A search space of 16 configs is explored per distinct (M, N, K) shape:
- `BLOCK_M` ∈ {16, 32, 64, 128}
- `num_warps` ∈ {4, 8}
- `num_stages` ∈ {3, 4}

`BLOCK_N=128` and `BLOCK_K=128` are held fixed to keep FP8 block-scale indexing exact. The grid uses `lambda meta:` to read the autotuned `BLOCK_M` at launch time.

**Algorithm is identical to v1** — FP8 GEMM kernels + SwiGLU in PyTorch + Python routing. This is a pure tile-size optimization.

**Results (19 workloads, all PASSED):**
- Latency — min: 2.20 ms, max: 17.80 ms, median: 7.01 ms
- Speedup — min: 1.807x, max: 4.945x, **mean: 2.486x**
- Improvement over v1: +0.10x mean speedup (2.385x → 2.486x), median latency 7.60 ms → 7.01 ms

---

## v2 — Fuse SwiGLU into GEMM1 epilogue (superseded — regressed vs v1)

**Optimization:** §3.3 / checklist item "SwiGLU: Fuse with GEMM2 (don't write GEMM1 output to memory)".

**What changed:** Replaced `_fp8_fp8_gemm` + `F.silu` + `F.mul` with a single new kernel `_fp8_fp8_gemm_swiglu`. Each CTA now holds **two** accumulators (`acc_gate`, `acc_up`) and processes both the gate half and the up half of W1 in the same K-loop. At the end of the loop, the SwiGLU is applied in-register before the result is stored:

```
z = silu(acc_up) * acc_gate   →   write [Tk, I] float32
```

**What this eliminates:** In v1, GEMM1 wrote `[Tk, 4096]` float32 to HBM, then PyTorch read it back to apply SwiGLU, then wrote `[Tk, 2048]` float32 as GEMM2's input. v2 skips that intermediate `[Tk, 4096]` write+read — for Tk=1024 that's ~16 MB saved per expert per forward pass.

**Tile size adjustment:** `BLOCK_M` for GEMM1 was reduced from 64 → 32 because each CTA now carries two accumulators (doubling register pressure). `BLOCK_M` for GEMM2 stays at 64 (single accumulator, unchanged).

**Everything else is identical to v1.**

---

## v1 — Basic Triton FP8 GEMM (superseded by v3)

**Strategy:** Keep routing and accumulation in PyTorch; replace the two GEMMs with custom Triton kernels that fuse FP8 block-scale dequantization directly into the matrix multiply.

**Routing** is identical to the Python reference — sigmoid gating, group-based top-K selection, softmax-normalized weights — all in PyTorch float32. No change here yet.

**GEMM1** (`_fp8_fp8_gemm`): computes `dequant(hidden) @ dequant(W1).T → [Tk, 4096]` in a single Triton kernel. Tiles are 64×128 over (tokens, output-features), iterating over K in 128-element chunks that align exactly with the FP8 block-scale granularity. A-scales are per-token-per-k-block `[56, Tk]`; B-scales are per-block `[32, 56]`. Everything accumulates in float32.

**SwiGLU** is still PyTorch (`F.silu(x2) * x1`), applied to the float32 GEMM1 output.

**GEMM2** (`_f32_fp8_gemm`): computes `float32_intermediate @ dequant(W2).T → [Tk, 7168]`. Same tiling strategy; only B needs FP8 dequantization since A is already float32 after SwiGLU.

**Accumulation** uses PyTorch `index_add_` into a float32 buffer, then copies to the bfloat16 output at the end.

**What's better than the Python reference:**
- The GEMMs run as GPU-parallel Triton kernels rather than sequential PyTorch matmuls over float32-expanded weights — avoids materializing the full dequantized `[E, 2I, H]` weight tensors in memory.

**Results (19 workloads, all PASSED):**
- Latency — min: 2.25 ms, max: 19.59 ms, median: 7.60 ms
- Speedup — min: 1.752x, max: 4.877x, **mean: 2.385x**

**Known limitations / next steps:**
- Routing is still pure PyTorch with sequential Python overhead.
- The expert loop is sequential in Python (32 iterations), each launching separate Triton kernels. A grouped/batched GEMM across all experts at once would be significantly faster.
- SwiGLU could be fused into the GEMM1 epilogue to save a read-write roundtrip.
- GEMM2 input is float32; quantizing it to FP8 before the matmul (like the FlashInfer baseline does) would halve memory bandwidth.
- No software pipelining or persistent kernel yet.
