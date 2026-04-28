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

    --baseline TEXT              Which baseline(s) to compare against.
                                 - "torch":      vs PyTorch reference only (fast,
                                                 works on any GPU).
                                 - "flashinfer": vs FlashInfer baseline only
                                                 (skips torch ref column).
                                 - "both":       both columns (default). On B200
                                                 this matches the official
                                                 evaluator. On H100 the flashinfer
                                                 path may not represent its
                                                 optimized target — relative
                                                 numbers are dev-only.
                                 Default: both
"""

from __future__ import annotations

import os
import re
import subprocess
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
    version_info: dict,
    max_workloads: int = None,
    debug: bool = False,
    profile: bool = False,
    baseline: str = "both",
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

    # ── Print run identity banner so logs make it obvious which version ran ─
    track = config["solution"].get("definition", "?")
    kver = version_info.get("kernel_version", "unknown")
    branch = version_info.get("branch", "unknown")
    commit = version_info.get("commit", "unknown")
    dirty = " (dirty)" if version_info.get("dirty") else ""
    print("=" * 70)
    print(f"  Kernel version : {kver}")
    print(f"  Definition     : {track}")
    print(f"  Git            : {branch}@{commit}{dirty}")
    print("=" * 70)

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

    # Optionally include the FlashInfer baseline solution shipped in the dataset
    # (e.g. flashinfer_wrapper_9sdjf3 for MoE). TraceSet.from_path already loaded
    # solutions/baseline/**/*.json into trace_set.solutions[definition].
    solutions_to_run = [solution]
    flashinfer_solution_name = None
    if baseline in ("flashinfer", "both"):
        existing = trace_set.solutions.get(definition, [])
        fi_sols = [s for s in existing if s.author == "flashinfer"]
        if not fi_sols:
            print(
                f"WARNING: --baseline={baseline} requested but no flashinfer baseline "
                f"found in dataset for {definition}. Falling back to torch ref only."
            )
            baseline = "torch"
        else:
            fi_sol = fi_sols[0]
            flashinfer_solution_name = fi_sol.name
            solutions_to_run.append(fi_sol)
            print(f"Including FlashInfer baseline: {fi_sol.name}")

    bench_ts = TraceSet(
        root=trace_set.root,
        definitions={definition: trace_set.definitions[definition]},
        solutions={definition: solutions_to_run},
        workloads={definition: workloads},
        traces={definition: []},
    )

    bench_cfg = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    result_ts = Benchmark(bench_ts, bench_cfg).run_all(dump_traces=True)

    # Pass 1: collect every trace keyed by (workload_uuid, solution_name) so we
    # can pair your kernel against each baseline.
    by_workload: dict = {}
    for trace in result_ts.traces.get(definition, []):
        if not trace.evaluation:
            continue
        # In flashinfer_bench traces, .workload and .solution are stored as
        # name/uuid strings, not the full objects.
        wl_obj = trace.workload
        wl_uuid = str(wl_obj.uuid) if hasattr(wl_obj, "uuid") else str(wl_obj)
        sol_obj = trace.solution
        sol_name = sol_obj.name if hasattr(sol_obj, "name") else str(sol_obj)
        entry = {
            "status": trace.evaluation.status.value,
            "log": trace.evaluation.log,
        }
        if trace.evaluation.performance:
            p = trace.evaluation.performance
            entry["latency_ms"] = p.latency_ms
            entry["reference_latency_ms"] = p.reference_latency_ms
            entry["speedup_vs_torch"] = p.speedup_factor
        if trace.evaluation.correctness:
            c = trace.evaluation.correctness
            entry["max_abs_error"] = c.max_absolute_error
            entry["max_rel_error"] = c.max_relative_error
        by_workload.setdefault(wl_uuid, {})[sol_name] = entry

    # Pass 2: build result rows for *your* solution; attach flashinfer latency
    # and derived speedup_vs_flashinfer when available.
    your_name = solution.name
    results = {}
    for wl_uuid, sols in by_workload.items():
        your = sols.get(your_name)
        if your is None:
            continue
        entry = dict(your)
        if flashinfer_solution_name and flashinfer_solution_name in sols:
            fi = sols[flashinfer_solution_name]
            fi_lat = fi.get("latency_ms")
            your_lat = your.get("latency_ms")
            entry["flashinfer_latency_ms"] = fi_lat
            entry["flashinfer_status"] = fi.get("status")
            if fi_lat is not None and your_lat is not None and your_lat > 0:
                entry["speedup_vs_flashinfer"] = fi_lat / your_lat
        results[wl_uuid] = entry

    # Print results inside the container too, so the speedup table is visible
    # even when run with --detach (where local main() exits before remote returns).
    print_results(results, baseline=baseline)

    return results


def _extract_kernel_version(sources: dict, entry_point: str) -> str:
    """Extract version string from the entry-point file's leading docstring.

    Matches `<anything> — v<digits><suffix>` on the first non-blank line of
    the docstring (e.g. `Triton FP8 Fused MoE kernel — v22`).  Returns
    "unknown" if no match.  This lets the version follow whatever the kernel
    author wrote in the docstring without any separate config to maintain.
    """
    entry_file = entry_point.split("::")[0]
    src = sources.get(entry_file)
    if src is None:
        return "unknown"
    # Look at first ~10 lines (docstring header) for "— vNN" pattern.
    for line in src.splitlines()[:10]:
        # em dash or hyphen, then 'v' + digits, optional suffix.
        m = re.search(r"[—\-]\s*(v\d+\w*)", line)
        if m:
            return m.group(1)
    return "unknown"


def _git_info() -> dict:
    """Capture git branch, short commit hash, and dirty state of the repo.

    Returns a dict with keys 'branch', 'commit', 'dirty' (and 'error' on
    failure).  Best-effort — never raises, so a missing git binary or a
    non-repo checkout just degrades to 'unknown'.
    """
    def _git(args: list[str]) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()

    try:
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        commit = _git(["rev-parse", "--short", "HEAD"])
        dirty = bool(_git(["status", "--porcelain"]))
        return {"branch": branch, "commit": commit, "dirty": dirty}
    except Exception as e:
        return {"branch": "unknown", "commit": "unknown", "dirty": False, "error": str(e)}


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


def print_results(results: dict, baseline: str = "both"):
    """Format and print the results dict returned from pack_and_run."""
    show_torch = baseline in ("torch", "both")
    show_fi = baseline in ("flashinfer", "both")

    headers = [f"{'Workload':<12}", f"{'Status':<14}", f"{'Latency (ms)':<14}"]
    if show_torch:
        headers.append(f"{'vs torch':<12}")
    if show_fi:
        headers.append(f"{'vs flashinfer':<16}")
    headers.append("abs_err")
    print("\n" + " ".join(headers))
    print("-" * sum(len(h) + 1 for h in headers))

    latencies, sp_torch, sp_fi = [], [], []
    for uuid, r in sorted(results.items()):
        status = r.get("status", "?")
        lat = r.get("latency_ms")
        st = r.get("speedup_vs_torch")
        sf = r.get("speedup_vs_flashinfer")
        abs_err = r.get("max_abs_error")

        lat_str = f"{lat:.4f}" if lat is not None else "N/A"
        st_str = f"{st:.3f}x" if st is not None else "N/A"
        sf_str = f"{sf:.3f}x" if sf is not None else "N/A"
        err_str = f"{abs_err:.2e}" if abs_err is not None else "N/A"

        cols = [f"{uuid[:8]:<12}", f"{status:<14}", f"{lat_str:<14}"]
        if show_torch:
            cols.append(f"{st_str:<12}")
        if show_fi:
            cols.append(f"{sf_str:<16}")
        cols.append(err_str)
        print(" ".join(cols))

        if lat is not None:
            latencies.append(lat)
        if st is not None:
            sp_torch.append(st)
        if sf is not None:
            sp_fi.append(sf)

    if latencies:
        import statistics
        print(f"\nSummary ({len(latencies)} workloads):")
        print(f"  Latency — min: {min(latencies):.4f} ms  max: {max(latencies):.4f} ms  median: {statistics.median(latencies):.4f} ms")
    if sp_torch:
        import statistics
        print(f"  vs torch       — min: {min(sp_torch):.3f}x  max: {max(sp_torch):.3f}x  mean: {statistics.mean(sp_torch):.3f}x")
    if sp_fi:
        import statistics
        print(f"  vs flashinfer  — min: {min(sp_fi):.3f}x  max: {max(sp_fi):.3f}x  mean: {statistics.mean(sp_fi):.3f}x")

    # Print log for *any* non-PASSED workload (NaN errors don't compare via >).
    # Sort by descending err with NaN last, take the first failure with a log.
    import math

    def _err_key(item):
        err = item[1].get("max_abs_error")
        if err is None or (isinstance(err, float) and math.isnan(err)):
            return -1.0
        return err

    fails = [
        (uuid, r) for uuid, r in results.items()
        if r.get("status") != "PASSED"
    ]
    fails.sort(key=_err_key, reverse=True)
    for uuid, r in fails:
        log = r.get("log") or "<no log captured>"
        err = r.get("max_abs_error")
        err_str = f"{err:.2e}" if err is not None else "N/A"
        print("\n" + "=" * 70)
        print(f"Failure log ({uuid[:8]}..., {r.get('status')}, abs_err={err_str}):")
        print("=" * 70)
        print(log)
        print("=" * 70)
        # Only print first to avoid flooding when many fails.
        break


@app.local_entrypoint()
def main(
    track: str,
    debug: bool = False,
    profile: bool = False,
    max_workloads: int = 0,
    baseline: str = "both",
):
    """Pack the solution for one track and run benchmark on Modal."""
    if baseline not in ("torch", "flashinfer", "both"):
        raise ValueError(f"--baseline must be torch|flashinfer|both, got {baseline!r}")

    print(f"Loading sources for track '{track}'...")
    sources, config = _load_track_sources(track)
    kernel_version = _extract_kernel_version(sources, config["build"]["entry_point"])
    git = _git_info()
    version_info = {"kernel_version": kernel_version, **git}
    dirty = " (dirty)" if git.get("dirty") else ""
    print(f"Kernel version: {kernel_version}  |  git: {git['branch']}@{git['commit']}{dirty}")
    print(f"Sending {len(sources)} file(s) to Modal: {list(sources.keys())}")
    print(f"Running on Modal {MODAL_GPU} (baseline={baseline})...")

    workload_limit = max_workloads if max_workloads and max_workloads > 0 else None
    results = pack_and_run.remote(
        sources,
        config,
        version_info,
        max_workloads=workload_limit,
        debug=debug,
        profile=profile,
        baseline=baseline,
    )

    if not results:
        print("No results returned!")
        return

    print_results(results, baseline=baseline)
