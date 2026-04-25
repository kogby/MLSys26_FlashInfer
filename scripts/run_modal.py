"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Usage:
    modal run scripts/run_modal.py [OPTIONS]

Options:
    --track TEXT                 [required] Track subdirectory (containing
                                 config.toml). One of:
                                 dsa_indexer | dsa_attention | moe.

    --use-official-baseline /    Load the upstream flashinfer+deep_gemm
    --no-use-official-baseline   baseline JSON instead of packing from source
                                 (sanity check; only meaningful for
                                 dsa_indexer).
                                 Default: --no-use-official-baseline

    --debug / --no-debug         Sets FIB_DEBUG=1/0 in the container. Kernels
                                 that support it (e.g. dsa_indexer) re-run the
                                 PyTorch reference and print a per-batch diff.
                                 Default: --no-debug

    --profile / --no-profile     Sets FIB_PROFILE=1/0 in the container.
                                 Kernels that support it print per-stage
                                 CUDA-event timings on every call.
                                 Default: --no-profile

    --max-workloads INT          Run only the first N workloads. 0 or negative
                                 means run all.
                                 Default: 0 (run all)

"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch", "triton", "numpy",
        # cupti-python: flashinfer-bench uses CUPTI for ~10ns precision kernel
        # timing (falls back to CUDA events if missing). Safe to keep — we're
        # not running any external CUPTI subscribers (NCU/nsys) anymore.
        "cupti-python",
        # Required by the dsa_attention track's kernel_cute_v2.py (imports
        # `cutlass.cute` for CuTe-DSL kernels). Harmless if unused by
        # dsa_indexer / moe tracks.
        "nvidia-cutlass-dsl",
        # Required ONLY when running --use-official-baseline (the dsa_indexer
        # baseline solution `flashinfer_deepgemm_wrapper_*.json` calls these at
        # runtime). Submission code must NOT import these — see FAQ.md L166-170.
        "flashinfer-python",
    )
    # IMPORTANT: install flashinfer-bench FROM SOURCE. The PyPI release is
    # older than 2026-04-10 and lacks the DsaTopkIndexerEvaluator (PR #354)
    # that handles NaN-ordering differences via sorted-score comparison.
    # FAQ.md recommends installing from source for latest eval changes.
    .run_commands(
        "git clone https://github.com/flashinfer-ai/flashinfer-bench.git /opt/flashinfer-bench",
        "cd /opt/flashinfer-bench && pip install -v -e .",
        # deep_gemm is not on PyPI; install from DeepSeek's GitHub. Only needed
        # for --use-official-baseline (same FAQ caveat as flashinfer above).
        # Disabled: upstream metadata-generation fails on Modal build. Re-enable
        # (and pin a known-good commit) when running --use-official-baseline.
        # "pip install git+https://github.com/deepseek-ai/DeepGEMM.git",
    )
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    max_workloads: int = None,
    debug: bool = False,
    profile: bool = False,
) -> dict:
    """Run benchmark on Modal B200 and return results.

    If `max_workloads` is set, only the first N workloads are run (handy for
    iterating on correctness / debugging before a full sweep).

    Generic env flags read by any track's kernel:
      FIB_DEBUG=1    → kernel may re-run the reference and print a diff
      FIB_PROFILE=1  → kernel may print per-stage CUDA-event timings
    """
    import os
    os.environ["FIB_DEBUG"] = "1" if debug else "0"
    os.environ["FIB_PROFILE"] = "1" if profile else "0"

    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    if max_workloads is not None:
        workloads = workloads[:max_workloads]
        print(f"DEBUG MODE: running only the first {len(workloads)} workloads")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
                "log": trace.evaluation.log,  # full stdout/stderr incl. compile errors
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    failure_logs_printed = False
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

        # If any workload failed, dump the log of the WORST failure (largest
        # abs_err) so we see the most revealing bug, not just the first mild one.
        if not failure_logs_printed:
            worst = None
            worst_err = -1.0
            for workload_uuid, result in traces.items():
                if result.get("status") != "PASSED" and result.get("log"):
                    err = result.get("max_abs_error") or 0
                    if err > worst_err:
                        worst_err = err
                        worst = (workload_uuid, result)
            if worst is not None:
                workload_uuid, result = worst
                print("\n" + "=" * 70)
                print(f"Worst failure log ({workload_uuid[:8]}..., "
                      f"{result.get('status')}, abs_err={worst_err:.2e}):")
                print("=" * 70)
                print(result["log"])
                print("=" * 70)
                failure_logs_printed = True


@app.local_entrypoint()
def main(
    track: str,
    use_official_baseline: bool = False,
    debug: bool = False,
    profile: bool = False,
    max_workloads: int = 0,
):
    """Pack the solution for one track and run benchmark on Modal.

    See the module-level docstring for the full option/example reference.
    """
    if use_official_baseline:
        baseline_path = (
            PROJECT_ROOT
            / "mlsys26-contest"
            / "solutions"
            / "baseline"
            / "dsa"
            / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
            / "flashinfer_deepgemm_wrapper_2ba145.json"
        )
        print(f"SANITY MODE: loading official baseline from {baseline_path}")
        solution = Solution.model_validate_json(baseline_path.read_text())
    else:
        from scripts.pack_solution import pack_solution
        print(f"Packing solution for track '{track}'...")
        solution_path = pack_solution(track)
        print("\nLoading solution...")
        solution = Solution.model_validate_json(solution_path.read_text())

    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    # Small subset while --profile is on — profile output would flood stdout
    # otherwise. Pass --max-workloads 0 for the full sweep.
    workload_limit = max_workloads if max_workloads and max_workloads > 0 else None
    results = run_benchmark.remote(
        solution,
        max_workloads=workload_limit,
        debug=debug,
        profile=profile,
    )

    if not results:
        print("No results returned!")
        return

    print_results(results)
