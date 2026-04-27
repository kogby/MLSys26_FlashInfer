"""
Run your Triton/CUDA kernel on Modal and get speedup vs FlashInfer baseline.

All flashinfer_bench work happens inside the Modal container (Linux). The local
side only reads source files and the per-track config.toml — no flashinfer_bench
import on macOS.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Usage:
    modal run scripts/run_modal.py --track <track> [OPTIONS]

    GPU defaults to B200:1; override with MODAL_GPU env var, e.g.:
        MODAL_GPU=H100:1 modal run scripts/run_modal.py --track moe

Options:
    --track TEXT                 [required] Track subdirectory (containing
                                 config.toml). Discovered dynamically — any
                                 immediate subdirectory of the project root
                                 with a config.toml works.

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

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import modal

# GPU is resolved at module-load time because @app.function decorators consume
# it eagerly. Set MODAL_GPU=H100:1 (or any spec Modal accepts) to override.
MODAL_GPU = os.environ.get("MODAL_GPU", "B200:1")

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        "flashinfer/flashinfer-ci-cu132:20260401-2c675fb",
        add_python="3.12",
    )
    .pip_install(
        # cupti-python: flashinfer-bench uses CUPTI for ~10ns precision kernel
        # timing (falls back to CUDA events if missing).
        "cupti-python",
        # Required for any CuTe-DSL kernel (e.g. moe GEMM2 rewrite).
        "nvidia-cutlass-dsl",
    )
    # Install flashinfer-bench FROM SOURCE — PyPI release lags main and lacks
    # evaluator fixes (e.g. DsaTopkIndexerEvaluator NaN-ordering, PR #354).
    .run_commands(
        "git clone https://github.com/flashinfer-ai/flashinfer-bench.git /opt/flashinfer-bench",
        "cd /opt/flashinfer-bench && pip install -v -e .",
    )
)


@app.function(image=image, gpu=MODAL_GPU, timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def pack_and_run(
    sources: dict,
    config: dict,
    max_workloads: int = None,
    debug: bool = False,
    profile: bool = False,
) -> dict:
    """Run entirely on the Modal Linux container.

    1. Writes source files to a temp directory.
    2. Packs them into a Solution via flashinfer_bench.
    3. Benchmarks against all workloads and returns a result dict.

    Generic env flags read by any track's kernel:
      FIB_DEBUG=1    → kernel may re-run the reference and print a diff
      FIB_PROFILE=1  → kernel may print per-stage CUDA-event timings
    """
    import os
    import tempfile

    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    os.environ["FIB_DEBUG"] = "1" if debug else "0"
    os.environ["FIB_PROFILE"] = "1" if profile else "0"

    sol_cfg = config["solution"]
    build_cfg = config["build"]

    language = build_cfg["language"]
    entry_point = build_cfg["entry_point"]
    dps = build_cfg.get("destination_passing_style", True)
    definition = sol_cfg["definition"]

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, content in sources.items():
            dest = os.path.join(tmpdir, filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as f:
                f.write(content)

        spec = BuildSpec(
            language=language,
            target_hardware=["cuda"],
            entry_point=entry_point,
            destination_passing_style=dps,
        )
        solution = pack_solution_from_files(
            path=tmpdir,
            spec=spec,
            name=sol_cfg["name"],
            definition=definition,
            author=sol_cfg["author"],
        )

    print(f"Packed: {solution.name}  (lang={language}, dps={dps})")

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    if definition not in trace_set.definitions:
        raise ValueError(f"Definition '{definition}' not found in trace set")

    workloads = trace_set.workloads.get(definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for '{definition}' in {TRACE_SET_PATH}")

    if max_workloads is not None and max_workloads > 0:
        workloads = workloads[:max_workloads]
        print(f"DEBUG MODE: running only the first {len(workloads)} workloads")

    bench_ts = TraceSet(
        root=trace_set.root,
        definitions={definition: trace_set.definitions[definition]},
        solutions={definition: [solution]},
        workloads={definition: workloads},
        traces={definition: []},
    )

    bench_cfg = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    result_ts = Benchmark(bench_ts, bench_cfg).run_all(dump_traces=True)

    results = {}
    for trace in result_ts.traces.get(definition, []):
        if not trace.evaluation:
            continue
        entry = {
            "status": trace.evaluation.status.value,
            "log": trace.evaluation.log,
        }
        if trace.evaluation.performance:
            p = trace.evaluation.performance
            entry["latency_ms"] = p.latency_ms
            entry["reference_latency_ms"] = p.reference_latency_ms
            entry["speedup_factor"] = p.speedup_factor
        if trace.evaluation.correctness:
            c = trace.evaluation.correctness
            entry["max_abs_error"] = c.max_absolute_error
            entry["max_rel_error"] = c.max_relative_error
        results[str(trace.workload.uuid)] = entry

    # Print results inside the container too, so the speedup table is visible
    # even when run with --detach (where local main() exits before remote returns).
    print_results(results)

    return results


def _discover_tracks() -> list[str]:
    """List immediate subdirs of PROJECT_ROOT that contain a config.toml."""
    return sorted(
        p.name for p in PROJECT_ROOT.iterdir()
        if p.is_dir() and (p / "config.toml").exists()
    )


def _load_track_sources(track: str) -> tuple[dict, dict]:
    """Read <track>/config.toml and gather source files. Local-only, no flashinfer-bench."""
    track_dir = PROJECT_ROOT / track
    config_path = track_dir / "config.toml"
    if not config_path.exists():
        available = _discover_tracks()
        raise FileNotFoundError(
            f"Track config not found: {config_path}. Available tracks: {available}"
        )

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    language = config["build"]["language"]
    if language == "triton":
        source_dir = track_dir / "solution" / "triton"
    elif language == "cuda":
        source_dir = track_dir / "solution" / "cuda"
    elif language == "python":
        source_dir = track_dir / "solution" / "python"
    else:
        raise ValueError(f"Unknown language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    sources = {f.name: f.read_text() for f in source_dir.glob("*.py")}
    if language == "cuda":
        sources.update({f.name: f.read_text() for f in source_dir.glob("*.cu")})
        sources.update({f.name: f.read_text() for f in source_dir.glob("*.cuh")})

    if not sources:
        raise FileNotFoundError(f"No source files found in {source_dir}")

    return sources, config


def print_results(results: dict):
    """Format and print the results dict returned from pack_and_run."""
    print(f"\n{'Workload':<12} {'Status':<14} {'Latency (ms)':<16} {'Speedup':<12} {'abs_err'}")
    print("-" * 72)

    latencies, speedups = [], []
    for uuid, r in sorted(results.items()):
        status = r.get("status", "?")
        lat = r.get("latency_ms")
        speedup = r.get("speedup_factor")
        abs_err = r.get("max_abs_error")

        lat_str = f"{lat:.4f}" if lat is not None else "N/A"
        speedup_str = f"{speedup:.3f}x" if speedup is not None else "N/A"
        err_str = f"{abs_err:.2e}" if abs_err is not None else "N/A"

        print(f"{uuid[:8]:<12} {status:<14} {lat_str:<16} {speedup_str:<12} {err_str}")

        if lat is not None:
            latencies.append(lat)
        if speedup is not None:
            speedups.append(speedup)

    if latencies:
        import statistics
        print(f"\nSummary ({len(latencies)} workloads):")
        print(f"  Latency — min: {min(latencies):.4f} ms  max: {max(latencies):.4f} ms  median: {statistics.median(latencies):.4f} ms")
    if speedups:
        import statistics
        print(f"  Speedup — min: {min(speedups):.3f}x  max: {max(speedups):.3f}x  mean: {statistics.mean(speedups):.3f}x")

    worst = None
    worst_err = -1.0
    for uuid, r in results.items():
        if r.get("status") != "PASSED" and r.get("log"):
            err = r.get("max_abs_error") or 0
            if err > worst_err:
                worst_err = err
                worst = (uuid, r)
    if worst is not None:
        uuid, r = worst
        print("\n" + "=" * 70)
        print(f"Worst failure log ({uuid[:8]}..., {r.get('status')}, abs_err={worst_err:.2e}):")
        print("=" * 70)
        print(r["log"])
        print("=" * 70)


@app.local_entrypoint()
def main(
    track: str,
    debug: bool = False,
    profile: bool = False,
    max_workloads: int = 0,
):
    """Pack the solution for one track and run benchmark on Modal."""
    print(f"Loading sources for track '{track}'...")
    sources, config = _load_track_sources(track)
    print(f"Sending {len(sources)} file(s) to Modal: {list(sources.keys())}")
    print(f"Running on Modal {MODAL_GPU}...")

    workload_limit = max_workloads if max_workloads and max_workloads > 0 else None
    results = pack_and_run.remote(
        sources,
        config,
        max_workloads=workload_limit,
        debug=debug,
        profile=profile,
    )

    if not results:
        print("No results returned!")
        return

    print_results(results)
