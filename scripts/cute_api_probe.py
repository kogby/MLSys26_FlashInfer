"""Probe nvidia-cutlass-dsl API surface on Modal B200.

Lists symbols available in cutlass.cute / cutlass.cute.nvgpu.tcgen05 / cpasync
so we know what Mma ops exist before writing the GEMM2 CuTe kernel.

Usage:
    modal run scripts/cute_api_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import modal

MODAL_GPU = os.environ.get("MODAL_GPU", "B200:1")

app = modal.App("flashinfer-cute-probe")

image = (
    modal.Image.from_registry(
        "flashinfer/flashinfer-ci-cu132:20260401-2c675fb",
        add_python="3.12",
    )
    .pip_install("nvidia-cutlass-dsl")
)


@app.function(image=image, gpu=MODAL_GPU, timeout=300)
def probe() -> dict:
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu as nvgpu
    from cutlass.cute.nvgpu import tcgen05, cpasync

    out = {}
    out["cutlass.__version__"] = getattr(cutlass, "__version__", "unknown")
    out["nvgpu_submodules"] = sorted(s for s in dir(nvgpu) if not s.startswith("_"))

    from cutlass.cute.nvgpu import warpgroup
    out["warpgroup_symbols"] = sorted(s for s in dir(warpgroup) if not s.startswith("_"))
    out["warpgroup_mma_ops"] = sorted(s for s in dir(warpgroup) if "Mma" in s)
    import inspect
    out["warpgroup_mma_signatures"] = {}
    for name in out["warpgroup_mma_ops"]:
        op_cls = getattr(warpgroup, name)
        try:
            sig = str(inspect.signature(op_cls.__init__))
        except (ValueError, TypeError):
            sig = "<unavailable>"
        out["warpgroup_mma_signatures"][name] = {
            "doc": (op_cls.__doc__ or "").splitlines()[:5],
            "init_sig": sig,
        }
    out["tcgen05_symbols"] = sorted(s for s in dir(tcgen05) if not s.startswith("_"))
    out["tcgen05_mma_ops"] = sorted(s for s in dir(tcgen05) if "Mma" in s)
    out["cpasync_symbols"] = sorted(s for s in dir(cpasync) if not s.startswith("_"))
    out["cute_dtypes"] = sorted(
        s for s in dir(cute) if not s.startswith("_") and (
            "Float" in s or "Int" in s or "BFloat" in s or "TFloat" in s
        )
    )
    # Inspect each Mma op's __init__ signature precisely.
    import inspect
    out["mma_signatures"] = {}
    for name in out["tcgen05_mma_ops"]:
        op_cls = getattr(tcgen05, name)
        try:
            sig = str(inspect.signature(op_cls.__init__))
        except (ValueError, TypeError):
            sig = "<signature unavailable>"
        out["mma_signatures"][name] = {
            "doc": (op_cls.__doc__ or "").splitlines()[:5],
            "init_sig": sig,
            "init_sig_doc": str(getattr(op_cls.__init__, "__doc__", "") or "")[:400],
        }
    return out


@app.local_entrypoint()
def main():
    print(f"Probing CuTe API on Modal {MODAL_GPU}...")
    info = probe.remote()
    print()
    print(f"cutlass version: {info['cutlass.__version__']}")
    print()
    print(f"nvgpu submodules ({len(info['nvgpu_submodules'])}):")
    for s in info["nvgpu_submodules"]:
        print(f"  {s}")
    print()
    print(f"warpgroup Mma ops ({len(info['warpgroup_mma_ops'])}):")
    for op in info["warpgroup_mma_ops"]:
        print(f"  - {op}")
    print()
    print(f"warpgroup all symbols ({len(info['warpgroup_symbols'])}):")
    for s in info["warpgroup_symbols"]:
        print(f"  {s}")
    print()
    print("warpgroup Mma signatures:")
    for name, sig in info["warpgroup_mma_signatures"].items():
        print(f"  {name}:")
        print(f"    sig: {sig['init_sig']}")
        for line in sig["doc"]:
            print(f"    | {line}")
    print()
    print(f"tcgen05 Mma ops ({len(info['tcgen05_mma_ops'])}):")
    for op in info["tcgen05_mma_ops"]:
        print(f"  - {op}")
    print()
    print(f"tcgen05 all symbols ({len(info['tcgen05_symbols'])}):")
    for s in info["tcgen05_symbols"]:
        print(f"  {s}")
    print()
    print(f"cpasync symbols ({len(info['cpasync_symbols'])}):")
    for s in info["cpasync_symbols"]:
        print(f"  {s}")
    print()
    print(f"cute dtype-ish symbols ({len(info['cute_dtypes'])}):")
    for s in info["cute_dtypes"]:
        print(f"  {s}")
    print()
    print("Mma op signatures:")
    for name, sig in info["mma_signatures"].items():
        print(f"  {name}:")
        print(f"    sig: {sig['init_sig']}")
        if sig["init_sig_doc"]:
            print(f"    doc: {sig['init_sig_doc']}")
