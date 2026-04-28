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
    import os
    import subprocess

    # Find cutlass-dsl install path via pip show then grep. cute is a namespace
    # package without a real __file__.
    cutlass_path = ""
    try:
        result = subprocess.run(
            ["pip", "show", "-f", "nvidia-cutlass-dsl"],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                cutlass_path = line.split(":", 1)[1].strip()
                break
    except Exception as e:
        cutlass_path = f"<err {e}>"
    out = {"cutlass_install_path": cutlass_path}

    # cuda-cutlass-dsl typically installs under site-packages/nvidia_cutlass_dsl
    # — search the whole tree.
    if cutlass_path and not cutlass_path.startswith("<"):
        # look for the actual cutlass code subdir
        candidates = [
            os.path.join(cutlass_path, "nvidia_cutlass_dsl", "python_packages"),
            os.path.join(cutlass_path, "cutlass"),
            cutlass_path,
        ]
        for c in candidates:
            if os.path.isdir(c):
                cutlass_path = c
                out["cutlass_install_path"] = c
                break

    # List cutlass examples / tests that mention warpgroup + epilogue
    try:
        result = subprocess.run(
            ["grep", "-rln", "--include=*.py",
             "warpgroup", cutlass_path],
            capture_output=True, text=True, timeout=30,
        )
        out["wgmma_files"] = result.stdout.strip().splitlines()[:20]
    except Exception as e:
        out["wgmma_files"] = f"<err {e}>"

    # Find epilogue examples
    try:
        result = subprocess.run(
            ["grep", "-rln", "--include=*.py",
             "make_fragment_C", cutlass_path],
            capture_output=True, text=True, timeout=30,
        )
        out["frag_C_files"] = result.stdout.strip().splitlines()[:20]
    except Exception as e:
        out["frag_C_files"] = f"<err {e}>"

    out = {}
    out["cutlass.__version__"] = getattr(cutlass, "__version__", "unknown")
    out["nvgpu_submodules"] = sorted(s for s in dir(nvgpu) if not s.startswith("_"))

    # ComposedLayout structure
    try:
        cl_cls = cute.typing.ComposedLayout
        out["ComposedLayout_attrs"] = sorted(
            s for s in dir(cl_cls) if not s.startswith("_")
        )
    except AttributeError:
        out["ComposedLayout_attrs"] = "<not found>"

    # Inspect hopper_helpers.make_smem_layout_a/b signatures
    import cutlass.utils.hopper_helpers as hh
    import inspect
    out["hopper_helpers_sigs"] = {}
    for name in (
        "make_smem_layout_a", "make_smem_layout_b",
        "make_smem_layout_atom", "make_trivial_tiled_mma",
        "get_smem_layout_atom",
    ):
        fn = getattr(hh, name, None)
        if fn is None:
            out["hopper_helpers_sigs"][name] = "<missing>"
            continue
        try:
            out["hopper_helpers_sigs"][name] = str(inspect.signature(fn))
        except (ValueError, TypeError):
            out["hopper_helpers_sigs"][name] = "<unavailable>"

    # Also peek LayoutEnum
    le = getattr(hh, "LayoutEnum", None)
    if le is not None:
        out["LayoutEnum_members"] = [s for s in dir(le) if not s.startswith("_")][:20]

    # cutlass.utils may have hopper_helpers (sm_90) similar to blackwell_helpers (sm_100).
    out["utils_submodules"] = sorted(s for s in dir(cutlass.utils) if not s.startswith("_"))
    for name in ("hopper_helpers", "sm90_utils", "wgmma_helpers", "blackwell_helpers"):
        try:
            mod = __import__(f"cutlass.utils.{name}", fromlist=[name])
            out[f"utils.{name}"] = sorted(s for s in dir(mod) if not s.startswith("_"))
        except ImportError:
            out[f"utils.{name}"] = "<not present>"

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
    print(f"cutlass install path: {info.get('cutlass_install_path')}")
    print()
    print("Files mentioning 'warpgroup' (for wgmma examples):")
    for f in info.get("wgmma_files", []):
        print(f"  {f}")
    print()
    print("Files using 'make_fragment_C' (for register-fragment epilogue):")
    for f in info.get("frag_C_files", []):
        print(f"  {f}")
    print()
    print(f"nvgpu submodules ({len(info['nvgpu_submodules'])}):")
    for s in info["nvgpu_submodules"]:
        print(f"  {s}")
    print()
    print()
    print(f"ComposedLayout attrs: {info.get('ComposedLayout_attrs')}")
    print()
    print("hopper_helpers signatures:")
    for name, sig in info.get("hopper_helpers_sigs", {}).items():
        print(f"  {name}: {sig}")
    if "LayoutEnum_members" in info:
        print(f"LayoutEnum members: {info['LayoutEnum_members']}")
    print()
    print(f"cutlass.utils submodules: {info['utils_submodules']}")
    for name in ("hopper_helpers", "sm90_utils", "wgmma_helpers", "blackwell_helpers"):
        v = info[f"utils.{name}"]
        if v == "<not present>":
            print(f"  utils.{name}: NOT PRESENT")
        else:
            print(f"  utils.{name} ({len(v)} symbols):")
            for s in v:
                print(f"    {s}")
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
