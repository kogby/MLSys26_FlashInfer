"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
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
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None, max_workloads: int = None) -> dict:
    """Run benchmark on Modal B200 and return results.

    If `max_workloads` is set, only the first N workloads are run (handy for
    iterating on correctness / debugging before a full sweep).
    """
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

        # If any workload failed, dump the log of the FIRST failure so we can
        # debug compile/runtime errors without flooding stdout with 80 copies.
        if not failure_logs_printed:
            for workload_uuid, result in traces.items():
                if result.get("status") != "PASSED" and result.get("log"):
                    print("\n" + "=" * 70)
                    print(f"First failure log ({workload_uuid[:8]}..., {result.get('status')}):")
                    print("=" * 70)
                    print(result["log"])
                    print("=" * 70)
                    failure_logs_printed = True
                    break


@app.local_entrypoint()
def main(use_official_baseline: bool = False):
    """Pack our solution and run benchmark on Modal.

    Args:
        use_official_baseline: if True, skip pack_solution and load the
            upstream flashinfer+deep_gemm baseline JSON instead. Used for
            sanity-checking that the framework runs cleanly end-to-end.
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
        print("Packing solution from source files...")
        solution_path = pack_solution()
        print("\nLoading solution...")
        solution = Solution.model_validate_json(solution_path.read_text())

    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    # For sanity-check runs, limit to a few workloads to keep iteration fast.
    max_workloads = 5 if use_official_baseline else None
    results = run_benchmark.remote(solution, max_workloads=max_workloads)

    if not results:
        print("No results returned!")
        return

    print_results(results)
