"""
CuTe hybrid dispatch — CuTe-1 (single kernel, full TOPK) for small T,
CuTe-2 (KV-split=8 + reduce) for large T.

Rationale (per-workload bench chart):
  CuTe-1: small-T 75.9× / large-T 27.8×   (one launch, no reduce overhead)
  CuTe-2: small-T 51.0× / large-T 45.8×   (more parallelism, reduce overhead)
Crossover near Small-T / Large-T cohort boundary.

Threshold tunable via env CUTE_T_THRESHOLD (default 8).

Loads sibling kernel_cute_v1.py / kernel_cute_v2.py by file path so this works
whether the harness imports as a package or a standalone file.
Both JIT-compile at import (~2× startup). OK for bench.
"""

import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str, fname: str):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, _HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_v1 = _load("kernel_cute_v1", "kernel_cute_v1.py")
_v2 = _load("kernel_cute_v2", "kernel_cute_v2.py")

T_THRESHOLD = int(os.environ.get("CUTE_T_THRESHOLD", "8"))
_LOG_T = os.environ.get("CUTE_LOG_T", "0") == "1"
_T_SEEN = {}   # T -> (count, path)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        sm_scale, output, lse):
    T = q_nope.shape[0]
    use_v1 = T <= T_THRESHOLD
    if _LOG_T:
        path = "v1" if use_v1 else "v2"
        c, _ = _T_SEEN.get(T, (0, path))
        _T_SEEN[T] = (c + 1, path)
        if c == 0:
            print(f"[cute_hybrid] T={T} -> {path} (threshold={T_THRESHOLD})", flush=True)
    impl = _v1 if use_v1 else _v2
    impl.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
             sm_scale, output, lse)


kernel = run
