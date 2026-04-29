#!/usr/bin/env python3
"""
add_fi_speedup.py

Reads FlashInfer baseline latencies from moe/solution/flashinfer-baseline/result.txt
and appends (or updates) a "Speedup (vs. FI-baseline)" summary line to each
triton result file in moe/solution/triton/result/*.txt.

Usage:
    # Process all result files
    python add_fi_speedup.py

    # Process specific files
    python add_fi_speedup.py result/v22.txt result/v21.txt

    # Dry-run (print what would be written, don't modify files)
    python add_fi_speedup.py --dry-run
"""

import argparse
import glob
import os
import re
import sys

# Path to the FlashInfer baseline result file, relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FI_BASELINE_FILE = os.path.join(
    SCRIPT_DIR, "..", "flashinfer-baseline", "result.txt"
)
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")

FI_LINE_TAG = "Speedup (vs. FI-baseline)"


def parse_fi_baseline(fi_file: str) -> dict[str, float]:
    """
    Parse FlashInfer baseline latencies.

    Expects lines like:
      1a4c6ba1         901  OK                 0.3160            0.5996            1.898x
    Returns {short_workload_id: latency_ms}.
    """
    latencies: dict[str, float] = {}
    with open(fi_file) as f:
        for line in f:
            # Match lines: <8-char-id>  <int>  OK  <float>  ...
            m = re.match(r"^\s*([0-9a-f]{8})\s+\d+\s+OK\s+([\d.]+)", line)
            if m:
                wid = m.group(1)
                lat = float(m.group(2))
                latencies[wid] = lat
    if not latencies:
        sys.exit(f"ERROR: No workload entries found in {fi_file}")
    return latencies


def parse_triton_result(result_file: str) -> dict[str, float]:
    """
    Parse triton result latencies.

    Expects lines like:
      1a4c6ba1     PASSED         0.5861           33.533x      2.99e+05
    Returns {short_workload_id: latency_ms}.
    """
    latencies: dict[str, float] = {}
    with open(result_file) as f:
        for line in f:
            m = re.match(r"^\s*([0-9a-f]{8})\s+\w+\s+([\d.]+)", line)
            if m:
                wid = m.group(1)
                lat = float(m.group(2))
                latencies[wid] = lat
    return latencies


def compute_speedups(
    fi_latencies: dict[str, float], triton_latencies: dict[str, float]
) -> list[float]:
    """
    Compute speedup = FI_latency / our_latency for each workload present in both dicts.
    A speedup > 1.0 means our kernel is faster than FlashInfer for that workload.
    """
    speedups = []
    for wid, our_lat in triton_latencies.items():
        if wid in fi_latencies and our_lat > 0:
            speedups.append(fi_latencies[wid] / our_lat)
    return speedups


def format_fi_line(speedups: list[float]) -> str:
    if not speedups:
        return f"  {FI_LINE_TAG} — (no matching workloads)"
    mn = min(speedups)
    mx = max(speedups)
    mean = sum(speedups) / len(speedups)
    return f"  {FI_LINE_TAG} — min: {mn:.3f}x  max: {mx:.3f}x  mean: {mean:.3f}x"


def update_result_file(result_file: str, new_line: str, dry_run: bool) -> None:
    with open(result_file) as f:
        content = f.read()

    # Remove any existing FI-baseline line
    lines = content.splitlines(keepends=True)
    lines = [l for l in lines if FI_LINE_TAG not in l]

    # Ensure file ends with a newline before appending
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    lines.append(new_line + "\n")
    new_content = "".join(lines)

    if dry_run:
        print(f"[DRY-RUN] Would update {result_file}:")
        print(f"  + {new_line}")
    else:
        with open(result_file, "w") as f:
            f.write(new_content)
        print(f"Updated {os.path.basename(result_file)}: {new_line.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "files",
        nargs="*",
        help="Result .txt files to update (default: all in result/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying any files",
    )
    parser.add_argument(
        "--fi-baseline",
        default=FI_BASELINE_FILE,
        help="Path to FlashInfer baseline result.txt",
    )
    args = parser.parse_args()

    # Resolve FlashInfer baseline
    fi_path = os.path.abspath(args.fi_baseline)
    if not os.path.isfile(fi_path):
        sys.exit(f"ERROR: FlashInfer baseline file not found: {fi_path}")
    fi_latencies = parse_fi_baseline(fi_path)
    print(f"Loaded FlashInfer baseline: {len(fi_latencies)} workloads from {fi_path}")

    # Resolve result files
    if args.files:
        result_files = [os.path.abspath(p) for p in args.files]
    else:
        result_files = sorted(glob.glob(os.path.join(RESULT_DIR, "v*.txt")))

    if not result_files:
        sys.exit(f"ERROR: No result files found in {RESULT_DIR}")

    print(f"Processing {len(result_files)} file(s)...\n")

    for rf in result_files:
        if not os.path.isfile(rf):
            print(f"WARNING: File not found, skipping: {rf}")
            continue

        triton_latencies = parse_triton_result(rf)
        if not triton_latencies:
            print(f"WARNING: No workload data found in {rf}, skipping")
            continue

        speedups = compute_speedups(fi_latencies, triton_latencies)
        if not speedups:
            print(f"WARNING: No matching workloads between FI baseline and {rf}")
            continue

        new_line = format_fi_line(speedups)
        update_result_file(rf, new_line, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
