"""Generate poster figures from REAL Modal B200 benchmark data."""
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# real_data is for sparse attention figs only; import lazily so indexer figs
# (which use indexer_data instead) work even without real_data.py present.

_IMAGES = Path(__file__).parent.parent / "images"
OUT_ATT   = _IMAGES / "dsa_att"     # K2 sparse attention figures
OUT_INDEX = _IMAGES / "dsa_index"   # K1 topk indexer figures
OUT_MOE   = _IMAGES / "moe"         # placeholder for MoE figures
for _d in (OUT_ATT, OUT_INDEX, OUT_MOE):
    _d.mkdir(parents=True, exist_ok=True)
# Back-compat alias for any helper that still references OUT (default to attention).
OUT = OUT_ATT

# When True, strip chart titles so the poster's own section headers aren't duplicated.
# Flip to False to regenerate standalone (report/slide) versions.
POSTER_MODE = True

# Shared size for the three charts in the bottom strip so they align.
STRIP_FIGSIZE = (8.0, 5.0)

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _set_title(ax, text):
    if not POSTER_MODE:
        ax.set_title(text)

CMU_RED = "#C41230"
NV_GREEN = "#76B900"
GRAY = "#5A5A5A"
LIGHT = "#BDBDBD"
DARK = "#2A2A2A"


def fig_speedup_bar():
    """Mean speedup per version (real data, 23 workloads each)."""
    from real_data import VERSIONS, stats
    order = ["v1", "v2", "v3", "v4", "cute_v1", "cute_v2"]
    labels = ["Triton v1\n(per-head grid)", "Triton v2\n(merged heads\n+ tl.dot)",
              "Triton v3\n(hybrid split)", "Triton v4\n(always KV-split)",
              "CuTe iter1\n(single-pass)", "CuTe iter2\n(warp-GEMV\n+ KV-split)"]
    means = [stats(VERSIONS[v])["mean_speedup"] for v in order]
    geos = [stats(VERSIONS[v])["geomean_speedup"] for v in order]
    colors = [LIGHT, GRAY, GRAY, DARK, NV_GREEN, CMU_RED]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=STRIP_FIGSIZE)
    bars = ax.bar(x, means, 0.65, color=colors, edgecolor="black", linewidth=0.8)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.8,
                f"{h:.1f}×", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup vs PyTorch reference")
    _set_title(ax, "Sparse Attention — Mean Speedup vs PyTorch  (B200, 23 workloads)")
    ax.set_ylim(0, max(means) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_speedup.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_small_vs_large():
    """Small-T vs large-T speedup per version."""
    from real_data import VERSIONS, stats
    order = ["v1", "v2", "v3", "v4", "cute_v1", "cute_v2"]
    labels = ["v1", "v2", "v3", "v4", "CuTe-1", "CuTe-2"]
    small = [stats(VERSIONS[v])["small_T_mean"] for v in order]
    large = [stats(VERSIONS[v])["large_T_mean"] for v in order]

    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=STRIP_FIGSIZE)
    ax.bar(x - w/2, small, w, color=NV_GREEN, edgecolor="black", label="Small-T workloads (9)")
    ax.bar(x + w/2, large, w, color=CMU_RED, edgecolor="black", label="Large-T workloads (14)")
    for i, (s, l) in enumerate(zip(small, large)):
        ax.text(i - w/2, s + 1, f"{s:.1f}×", ha="center", fontsize=11, fontweight="bold")
        ax.text(i + w/2, l + 1, f"{l:.1f}×", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup vs PyTorch")
    _set_title(ax, "Small-T vs Large-T Speedup — Decode Regime Behavior")
    ax.legend(framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(small), max(large)) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT / "fig_small_vs_large.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_latency():
    """Mean latency (ms) for small-T and large-T workload groups."""
    from real_data import VERSIONS, stats
    order = ["v1", "v2", "v3", "v4", "cute_v1", "cute_v2"]
    labels = ["v1", "v2", "v3", "v4", "CuTe-1", "CuTe-2"]
    smlat = [stats(VERSIONS[v])["small_T_lat"] for v in order]
    lglat = [stats(VERSIONS[v])["large_T_lat"] for v in order]

    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, smlat, w, color=NV_GREEN, edgecolor="black", label="Small-T (9 workloads)")
    ax.bar(x + w/2, lglat, w, color=CMU_RED, edgecolor="black", label="Large-T (14 workloads)")
    for i, (s, l) in enumerate(zip(smlat, lglat)):
        ax.text(i - w/2, s * 1.15, f"{s*1000:.0f} µs", ha="center", fontsize=11, fontweight="bold")
        ax.text(i + w/2, l * 1.15, f"{l*1000:.0f} µs", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Kernel Latency — Small-T vs Large-T Workloads")
    ax.set_yscale("log")
    ax.legend(framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_per_workload():
    """Per-workload scatter: speedup across 23 workloads for best 3 versions."""
    from real_data import VERSIONS
    fig, ax = plt.subplots(figsize=STRIP_FIGSIZE)
    wid = np.arange(23)
    for v, color, marker, label in [
        ("v4", DARK, "o", "Triton v4"),
        ("cute_v1", NV_GREEN, "s", "CuTe iter1"),
        ("cute_v2", CMU_RED, "^", "CuTe iter2 (best)"),
    ]:
        sp = [x[1] for x in VERSIONS[v]]
        ax.plot(wid, sp, marker=marker, color=color, linewidth=1.8, markersize=9, label=label)
    ax.axvline(8.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(4, ax.get_ylim()[1] * 0.92, "Small-T (9)", ha="center", fontsize=13, color="gray")
    ax.text(15.5, ax.get_ylim()[1] * 0.92, "Large-T (14)", ha="center", fontsize=13, color="gray")
    ax.set_xlabel("Workload index (0–22)")
    ax.set_ylabel("Speedup vs PyTorch")
    _set_title(ax, "Per-Workload Speedup (23 configs)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_workload.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_sm_occupancy():
    """Wave-1 block coverage at T=1 decode across three kernel grid designs.

    Shows blocks_dispatched / num_SMs — NOT measured SM utilisation.
    Real SM utilisation depends on warps/block, regs, smem, memory stalls,
    and would need NCU's sm__warps_active metric. This figure only shows
    how many SMs receive at least one block in the first wave.

    Grids (verified in source):
      v2 / v4 small-T path: (T,)         → T=1 :   1 block
      v4 KV-split path:     (T, 8)       → T=1 :   8 blocks
      cute_v2:              (T, H, 8)    → T=1 : 128 blocks   (per-head × split)
    B200 SM count: 148 (NVIDIA spec).
    """
    # 148 SMs: 8 rows × 18 cols = 144, pad 4 → lay out as 8×19 with 4 blanks.
    # Simpler: 4×37 doesn't read well; use 8×19 and mask the extra 4 cells.
    rows, cols = 8, 19
    total_slots = rows * cols  # 152
    total_sm = 148
    configs = [
        ("Triton v2  grid (T,)\n1 block dispatched", 1, LIGHT),
        ("Triton v4 KV-split  grid (T, 8)\n8 blocks dispatched", 8, NV_GREEN),
        ("CuTe v2  grid (T, H, 8)\n128 blocks dispatched", 128, CMU_RED),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    from matplotlib.colors import ListedColormap
    for ax, (title, n_busy, color) in zip(axes, configs):
        # -1 = not-an-SM slot (padding), 0 = idle SM, 1 = covered SM
        grid = np.zeros((rows, cols))
        # last 4 cells of grid are padding (not SMs) → mark -1
        grid.flat[total_sm:] = -1
        covered = min(n_busy, total_sm)
        idx = np.linspace(0, total_sm - 1, covered, dtype=int)
        grid.flat[idx] = 1
        cmap = ListedColormap(["white", "#F0F0F0", color])
        ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
        # white gridlines between cells
        for r in range(rows + 1):
            ax.axhline(r - 0.5, color="white", lw=0.8)
        for c in range(cols + 1):
            ax.axvline(c - 0.5, color="white", lw=0.8)
        pct = covered / total_sm * 100
        ax.set_title(
            f"{title}\n→ {pct:.0f}% of SMs receive a block (wave 1)",
            fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor("black")
    # shared footnote clarifying the metric
    fig.text(0.5, -0.02,
             "Illustrative: wave-1 block→SM coverage, not measured SM utilisation.  "
             "B200 = 148 SMs.",
             ha="center", fontsize=10, style="italic", color=DARK)
    if not POSTER_MODE:
        fig.suptitle("Decode Parallelism at T=1  —  Block → SM Coverage",
                     fontsize=16, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sm_occupancy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_hbm_traffic():
    labels = ["2-pass\n(baseline)", "Single-pass\n(online softmax)"]
    values = [512, 256]
    colors = [GRAY, CMU_RED]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{v} KB",
                ha="center", fontweight="bold", fontsize=16)
    ax.set_ylabel("K-cache HBM read per split (KB)")
    _set_title(ax, "K-Cache Bandwidth — 2× Reduction")
    ax.set_ylim(0, 600)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hbm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_kv_broadcast():
    """v1 per-head vs v2 merged-heads — real byte count for MLA KV cache.

    Per-token MLA KV:
      ckv 512 dims + kpe 64 dims = 576 elems × 2 B (bf16) = 1,152 B
    Per 2048-topk KV window:
      2048 × 1,152 B = 2,359,296 B ≈ 2.25 MiB  (baseline, loaded once)

    v1 grid [T, H]: each of 16 heads reloads the same window → 16× traffic.
    v2 grid [T]:    one program loads once, broadcasts across heads → 1×.
    """
    MIB = 1024 * 1024
    per_window = 2048 * (512 + 64) * 2  # bytes, bf16
    v1_bytes = 16 * per_window
    v2_bytes = per_window

    labels = ["v1 per-head grid\n(16 heads reload KV)",
              "v2 merged-heads grid\n(1 program, 16 heads)"]
    values_mib = [v1_bytes / MIB, v2_bytes / MIB]
    colors = [GRAY, NV_GREEN]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values_mib, color=colors, edgecolor="black")
    for b, v in zip(bars, values_mib):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                f"{v:.1f} MiB", ha="center", fontweight="bold", fontsize=15)
    ax.set_ylabel("K-cache HBM read per token (MiB)")
    ax.set_ylim(0, max(values_mib) * 1.38)
    ax.grid(axis="y", alpha=0.3)
    # annotated arrow between the two bars, well above both value labels
    y_anno = max(values_mib) * 1.22
    ax.annotate("", xy=(1, y_anno), xytext=(0, y_anno),
                arrowprops=dict(arrowstyle="->", color=CMU_RED, lw=2))
    ax.text(0.5, y_anno * 1.04,
            f"{v1_bytes / v2_bytes:.0f}× reduction",
            ha="center", va="bottom", fontsize=15, fontweight="bold", color=CMU_RED)
    ax.text(0.5, -0.18,
            "bf16 · TOPK = 2048 · 512 ckv + 64 kpe dims",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=11, style="italic", color=DARK)
    _set_title(ax, "KV-Cache Broadcast — v1 per-head vs v2 merged (MLA, bf16)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_kv_broadcast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── TopK Indexer figures ─────────────────────────────────────────────────────


def fig_indexer_speedup_by_size():
    """Indexer mean speedup across the 4 batch-size classes (real B200 data).

    Style matches K2 sparse-attention bars: gradient from light → dark → red
    to convey "harder workloads → bigger win" narrative.
    """
    from indexer_data import SIZE_BOUNDS, indexer_stats
    s = indexer_stats()

    classes = ["Small-T", "Medium-T", "Large-T", "XLarge-T"]
    means = [s[c]["mean"] for c in classes]
    mins = [s[c]["min"] for c in classes]
    maxs = [s[c]["max"] for c in classes]
    # Class definitions (batch_size ranges) measured from the workload set:
    # Small-T  B=1–4, Medium-T B=6–8, Large-T B=11–16, XLarge-T B=25–31.
    bs_ranges = {
        "Small-T":  "B = 1–4",
        "Medium-T": "B = 6–8",
        "Large-T":  "B = 11–16",
        "XLarge-T": "B = 25–31",
    }
    # Same gradient convention as K2 fig_speedup_bar
    colors = [LIGHT, GRAY, DARK, CMU_RED]

    labels = [f"{c}\n({bs_ranges[c]})" for c in classes]

    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bars = ax.bar(x, means, 0.65, color=colors, edgecolor="black", linewidth=0.8)
    # Range whiskers (min–max), drawn first so the value labels above sit on top
    for i, (lo, hi) in enumerate(zip(mins, maxs)):
        ax.plot([i, i], [lo, hi], color="black", lw=1.2, zorder=2)
        ax.plot([i - 0.08, i + 0.08], [lo, lo], color="black", lw=1.2, zorder=2)
        ax.plot([i - 0.08, i + 0.08], [hi, hi], color="black", lw=1.2, zorder=2)
    # Place the bold mean label above the upper whisker so they don't overlap
    for i, (m, hi) in enumerate(zip(means, maxs)):
        ax.text(i, hi + max(maxs) * 0.025,
                f"{m:.1f}×", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup vs PyTorch reference")
    _set_title(ax, f"TopK Indexer — Mean Speedup by Workload Class  "
                   f"(B200, {s['all']['n']} workloads)")
    ax.set_ylim(0, max(maxs) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_INDEX / "fig_indexer_speedup.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_INDEX / "fig_indexer_speedup.svg", bbox_inches="tight")
    plt.close(fig)


def fig_indexer_versions():
    """TopK indexer optimization timeline — mean speedup per version.

    Style mirrors the K2 sparse-attention bar chart so both posters read alike.
    Data sourced from notes/topk_indexer_level1.md progress table; the final
    bar uses the latest 128-workload sweep (NVMLE).
    """
    from indexer_data import INDEXER_VERSIONS

    labels = [lbl for lbl, _ in INDEXER_VERSIONS]
    means = [m for _, m in INDEXER_VERSIONS]
    # Light → dark → red gradient (final bar = CMU red highlight)
    colors = [LIGHT, GRAY, DARK, CMU_RED]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bars = ax.bar(x, means, 0.65, color=colors, edgecolor="black", linewidth=0.8)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(means) * 0.02,
                f"{m:.2f}×", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup vs PyTorch reference")
    _set_title(ax, "TopK Indexer — Mean Speedup vs PyTorch  (B200)")
    ax.set_ylim(0, max(means) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_INDEX / "fig_indexer_versions.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_INDEX / "fig_indexer_versions.svg", bbox_inches="tight")
    plt.close(fig)


def fig_indexer_per_workload():
    """Per-workload speedup line chart (single colour, points connected)."""
    from indexer_data import INDEXER_RESULTS, SIZE_BOUNDS

    speedups = [s for _, s in INDEXER_RESULTS]
    n = len(speedups)
    wid = np.arange(n)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(wid, speedups, marker="o", color=CMU_RED, linewidth=1.4,
            markersize=3.5, markeredgecolor="black", markeredgewidth=0.3,
            label="TopK Indexer (Triton FP8)", zorder=3)

    # Class boundary dashed lines + labels
    classes = ["Small-T", "Medium-T", "Large-T", "XLarge-T"]
    boundaries = [SIZE_BOUNDS[c][1] - 0.5 for c in classes[:-1]]
    for b in boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.4, zorder=1)

    bs_ranges = {
        "Small-T":  "B = 1–4",
        "Medium-T": "B = 6–8",
        "Large-T":  "B = 11–16",
        "XLarge-T": "B = 25–31",
    }
    ymax = max(speedups)
    # Reserve a 22% headroom strip above the data so labels never collide
    # with the line. Labels sit centred in that strip.
    ax.set_ylim(0, ymax * 1.22)
    label_y = ymax * 1.13
    for c in classes:
        lo, hi = SIZE_BOUNDS[c]
        hi = min(hi, n)
        mid = (lo + hi - 1) / 2
        ax.text(mid, label_y, f"{c}\n{bs_ranges[c]}", ha="center", va="center",
                fontsize=11, color="gray", style="italic")

    ax.set_xlabel("Workload index")
    ax.set_ylabel("Speedup vs PyTorch")
    _set_title(ax, f"TopK Indexer — Per-Workload Speedup ({n} configs)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT_INDEX / "fig_indexer_per_workload.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_INDEX / "fig_indexer_per_workload.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_speedup_bar()
    fig_small_vs_large()
    fig_latency()
    fig_per_workload()
    fig_sm_occupancy()
    fig_hbm_traffic()
    fig_kv_broadcast()
    fig_indexer_speedup_by_size()
    fig_indexer_per_workload()
    print("Saved all figures to", OUT)
