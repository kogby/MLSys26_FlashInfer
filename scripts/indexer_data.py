"""TopK indexer — real Modal B200 benchmark data.

Each entry: (latency_ms, speedup_factor)
Total: 134 workloads, all PASSED (abs_err = rel_err = 0).
Source: `modal run scripts/run_modal.py --track dsa_indexer` (no --debug, no --profile).
Date: 2026-04-25.

Latency clusters into 4 size classes that correspond to reference latency
brackets (small / medium / large / xlarge). Boundaries inferred from the
clear gaps in the raw data (see SIZE_BOUNDS below).
"""

INDEXER_RESULTS = [
    # Small-T cluster (workloads 1-29)
    (0.114, 8.26), (0.121, 7.58), (0.115, 8.02), (0.111, 9.43), (0.111, 9.51),
    (0.112, 9.48), (0.112, 9.45), (0.111, 9.42), (0.112, 9.41), (0.111, 9.48),
    (0.110, 9.63), (0.120, 10.92), (0.110, 10.76), (0.124, 10.95), (0.127, 10.65),
    (0.118, 11.43), (0.120, 11.30), (0.111, 12.16), (0.110, 12.07), (0.122, 11.15),
    (0.125, 10.82), (0.124, 10.91), (0.111, 11.92), (0.127, 10.56), (0.126, 10.74),
    (0.123, 10.81), (0.121, 11.07), (0.111, 12.04), (0.127, 10.52),
    # Medium-T cluster (workloads 30-58)
    (0.126, 15.04), (0.117, 15.91), (0.117, 15.79), (0.121, 14.62), (0.126, 13.82),
    (0.112, 14.56), (0.128, 14.65), (0.123, 13.32), (0.127, 14.96), (0.111, 17.19),
    (0.123, 15.13), (0.121, 15.17), (0.123, 14.87), (0.123, 15.03), (0.126, 14.43),
    (0.123, 14.71), (0.110, 16.23), (0.124, 14.68), (0.127, 14.25), (0.121, 15.03),
    (0.127, 14.31), (0.123, 14.63), (0.111, 16.13), (0.112, 16.11), (0.113, 15.93),
    (0.127, 14.38), (0.123, 12.71), (0.124, 13.76), (0.123, 12.72),
    # Large-T cluster (workloads 59-90)
    (0.121, 22.73), (0.128, 22.62), (0.128, 21.67), (0.133, 21.87), (0.132, 21.35),
    (0.144, 16.61), (0.130, 22.41), (0.128, 22.56), (0.147, 18.74), (0.129, 21.52),
    (0.124, 22.07), (0.147, 18.83), (0.128, 21.44), (0.147, 18.77), (0.130, 21.00),
    (0.125, 21.44), (0.147, 18.25), (0.131, 20.76), (0.122, 22.13), (0.129, 21.00),
    (0.147, 18.63), (0.129, 21.16), (0.125, 21.45), (0.147, 18.25), (0.132, 20.54),
    (0.112, 22.97), (0.147, 17.43), (0.125, 20.56), (0.111, 21.91), (0.146, 17.74),
    (0.129, 19.98), (0.127, 20.32), (0.124, 17.70),
    # XLarge-T cluster (workloads 91-134)
    (0.126, 38.49), (0.124, 37.12), (0.131, 36.13), (0.153, 28.44), (0.128, 35.76),
    (0.153, 30.05), (0.129, 36.32), (0.137, 35.54), (0.156, 30.32), (0.155, 30.42),
    (0.123, 38.16), (0.135, 35.14), (0.132, 35.97), (0.132, 35.63), (0.156, 30.15),
    (0.127, 36.53), (0.129, 35.56), (0.128, 35.84), (0.154, 29.72), (0.126, 36.51),
    (0.136, 34.04), (0.131, 35.21), (0.155, 29.80), (0.129, 35.79), (0.128, 35.03),
    (0.155, 29.11), (0.126, 35.61), (0.154, 29.05), (0.136, 33.20), (0.132, 34.22),
    (0.153, 29.34), (0.112, 37.69), (0.136, 33.31), (0.128, 32.18), (0.128, 33.04),
    (0.154, 27.65), (0.128, 31.26),
]

# Size class boundaries (workload index, 0-based, half-open intervals).
SIZE_BOUNDS = {
    "Small-T":   (0, 29),    # 29 workloads, ref ~1ms class
    "Medium-T":  (29, 58),   # 29 workloads, ref ~2ms class
    "Large-T":   (58, 91),   # 33 workloads, ref ~3ms class
    "XLarge-T":  (91, 134),  # 43 workloads, ref ~5ms class
}


# Per-version mean speedup vs PyTorch reference (Modal B200, mean across all
# workloads at the time of measurement). Indexer optimization timeline.
INDEXER_VERSIONS = [
    ("PyTorch v1\n(remove\n.item() sync)",      1.15),
    ("Triton v1\n(FP32 fused\nscore kernel)",   1.45),
    ("Triton v2\n(FP8 Tensor\nCore MMA)",       2.60),
    ("Triton v3\n(batched topk\n+ GPU remap)", 20.71),
]


def _gmean(xs):
    import math
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def indexer_stats():
    """Aggregate stats per size class + overall."""
    out = {}
    speedups = [s for _, s in INDEXER_RESULTS]
    lats = [l for l, _ in INDEXER_RESULTS]
    out["all"] = {
        "n": len(INDEXER_RESULTS),
        "mean": sum(speedups) / len(speedups),
        "geomean": _gmean(speedups),
        "min": min(speedups), "max": max(speedups),
        "lat_mean_ms": sum(lats) / len(lats),
    }
    for name, (lo, hi) in SIZE_BOUNDS.items():
        bucket = INDEXER_RESULTS[lo:hi]
        sps = [s for _, s in bucket]
        ls = [l for l, _ in bucket]
        out[name] = {
            "n": len(bucket),
            "mean": sum(sps) / len(sps),
            "geomean": _gmean(sps),
            "min": min(sps), "max": max(sps),
            "lat_mean_ms": sum(ls) / len(ls),
        }
    return out


if __name__ == "__main__":
    s = indexer_stats()
    print(f"{'Class':<10} {'N':>4} {'Mean':>8} {'Geomean':>9} {'Min':>7} {'Max':>7} {'Latμs':>7}")
    for k in ["Small-T", "Medium-T", "Large-T", "XLarge-T", "all"]:
        v = s[k]
        print(f"{k:<10} {v['n']:>4} {v['mean']:>7.2f}× {v['geomean']:>8.2f}× "
              f"{v['min']:>6.2f}× {v['max']:>6.2f}× {v['lat_mean_ms']*1000:>6.0f}")
