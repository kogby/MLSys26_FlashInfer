#!/usr/bin/env python3
"""
plot_progress.py

Plots kernel optimization progress across versions, showing mean speedup
vs. PyTorch baseline (right y-axis) and vs. FlashInfer baseline (left y-axis).

Usage:
    python plot_progress.py                  # saves progress.png next to this script
    python plot_progress.py --out my.png     # custom output path
    python plot_progress.py --fi-only        # only plot FI speedup (single axis)
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")

# ── parsing ───────────────────────────────────────────────────────────────────

def parse_result_files(result_dir: str) -> list[dict]:
    """
    Parse all v*.txt result files and return a list of dicts sorted by version.
    Each dict: {version, pt_mean, fi_mean}  (None if the line is absent)
    """
    records = []
    for path in sorted(glob.glob(os.path.join(result_dir, "v*.txt"))):
        fname = os.path.basename(path)
        m = re.match(r"v(\d+)\.txt$", fname)
        if not m:
            continue
        version = int(m.group(1))

        pt_mean = fi_mean = None
        with open(path) as f:
            for line in f:
                m2 = re.search(r"Speedup \(vs\. PyTorch\).*mean:\s*([\d.]+)x", line)
                if m2:
                    pt_mean = float(m2.group(1))
                m3 = re.search(r"Speedup \(vs\. FI-baseline\).*mean:\s*([\d.]+)x", line)
                if m3:
                    fi_mean = float(m3.group(1))

        records.append({"version": version, "pt_mean": pt_mean, "fi_mean": fi_mean})

    records.sort(key=lambda r: r["version"])
    return records


# ── plotting ──────────────────────────────────────────────────────────────────

COLOR_FI = "#2196F3"   # blue  — FI speedup (left axis)
COLOR_PT = "#FF5722"   # orange — PyTorch speedup (right axis)
COLOR_REF = "#4CAF50"  # green  — FI=1.0 reference


def plot_dual(records: list[dict], out_path: str) -> None:
    versions = [r["version"] for r in records]
    labels   = [f"v{v}" for v in versions]
    x        = np.arange(len(versions))

    fi_vals = [r["fi_mean"] for r in records]
    pt_vals = [r["pt_mean"] for r in records]

    fig, ax_fi = plt.subplots(figsize=(11, 5))
    ax_pt = ax_fi.twinx()

    # ── FI speedup line (left axis) ──
    ax_fi.plot(x, fi_vals, color=COLOR_FI, marker="o", linewidth=2.2,
               markersize=6, label="vs. FlashInfer (left axis)", zorder=3)
    ax_fi.axhline(1.0, color=COLOR_REF, linewidth=1.2, linestyle="--",
                  label="FI baseline (1.0×)", zorder=2)

    # ── PyTorch speedup line (right axis) ──
    ax_pt.plot(x, pt_vals, color=COLOR_PT, marker="s", linewidth=2.2,
               markersize=6, label="vs. PyTorch (right axis)", zorder=3)

    # ── data labels ──
    for xi, fv, pv in zip(x, fi_vals, pt_vals):
        ax_fi.annotate(f"{fv:.2f}×", (xi, fv),
                       textcoords="offset points", xytext=(0, 8),
                       ha="center", fontsize=7.5, color=COLOR_FI)
        ax_pt.annotate(f"{pv:.0f}×", (xi, pv),
                       textcoords="offset points", xytext=(0, -14),
                       ha="center", fontsize=7.5, color=COLOR_PT)

    # ── axes formatting ──
    ax_fi.set_xticks(x)
    ax_fi.set_xticklabels(labels, fontsize=10)
    ax_fi.set_xlabel("Kernel version", fontsize=11)

    ax_fi.set_ylabel("Speedup vs. FlashInfer  (↑ better)", color=COLOR_FI, fontsize=11)
    ax_fi.tick_params(axis="y", labelcolor=COLOR_FI)
    ax_fi.set_ylim(0, max(v for v in fi_vals if v) * 1.35)
    ax_fi.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f×"))

    ax_pt.set_ylabel("Speedup vs. PyTorch  (↑ better)", color=COLOR_PT, fontsize=11)
    ax_pt.tick_params(axis="y", labelcolor=COLOR_PT)
    ax_pt.set_ylim(0, max(v for v in pt_vals if v) * 1.25)
    ax_pt.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f×"))

    # ── legend — merge both axes ──
    lines_fi, labs_fi = ax_fi.get_legend_handles_labels()
    lines_pt, labs_pt = ax_pt.get_legend_handles_labels()
    ax_fi.legend(lines_fi + lines_pt, labs_fi + labs_pt,
                 loc="upper left", fontsize=9, framealpha=0.85)

    ax_fi.set_title("Fused-MoE Triton kernel — optimization progress",
                    fontsize=13, fontweight="bold", pad=10)
    ax_fi.grid(axis="y", linestyle=":", alpha=0.4)
    ax_fi.set_xlim(-0.5, len(x) - 0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def plot_fi_only(records: list[dict], out_path: str) -> None:
    versions = [r["version"] for r in records]
    labels   = [f"v{v}" for v in versions]
    x        = np.arange(len(versions))
    fi_vals  = [r["fi_mean"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(x, fi_vals, color=COLOR_FI, marker="o", linewidth=2.2,
            markersize=7, zorder=3)
    ax.axhline(1.0, color=COLOR_REF, linewidth=1.3, linestyle="--",
               label="FlashInfer baseline (1.0×)")

    for xi, fv in zip(x, fi_vals):
        ax.annotate(f"{fv:.3f}×", (xi, fv),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=COLOR_FI)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Kernel version", fontsize=11)
    ax.set_ylabel("Speedup vs. FlashInfer  (↑ better)", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f×"))
    ax.set_ylim(0, max(fi_vals) * 1.35)
    ax.legend(fontsize=9)
    ax.set_title("Fused-MoE Triton kernel — speedup vs. FlashInfer baseline",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_xlim(-0.5, len(x) - 0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "progress.png"),
                        help="Output image path (default: progress.png next to this script)")
    parser.add_argument("--fi-only", action="store_true",
                        help="Only plot speedup vs. FlashInfer (single axis)")
    args = parser.parse_args()

    records = parse_result_files(RESULT_DIR)
    if not records:
        sys.exit(f"ERROR: no v*.txt files found in {RESULT_DIR}")

    # Filter out records that are missing the required speedup values
    if args.fi_only:
        records = [r for r in records if r["fi_mean"] is not None]
        plot_fi_only(records, args.out)
    else:
        records = [r for r in records if r["fi_mean"] is not None
                                      and r["pt_mean"] is not None]
        plot_dual(records, args.out)


if __name__ == "__main__":
    main()
