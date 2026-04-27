"""
PyTorch profiler for a track's kernel on Modal. Writes a Chrome trace JSON to
the flashinfer-trace volume; download and open in https://ui.perfetto.dev or
chrome://tracing.

This is a single-workload profiler — for benchmarking against the FlashInfer
baseline use scripts/run_modal.py.

Usage:
    modal run scripts/profile_kernel.py --track moe

    GPU defaults to B200:1; override with MODAL_GPU env var:
        MODAL_GPU=H100:1 modal run scripts/profile_kernel.py --track moe

After the run:
    modal volume get flashinfer-trace kernel_profile.json
    # drag kernel_profile.json into https://ui.perfetto.dev
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

MODAL_GPU = os.environ.get("MODAL_GPU", "B200:1")

app = modal.App("flashinfer-profile")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_PATH = "/data"

# Match run_modal.py's image: from-source flashinfer-bench is overkill for
# profiling but keeps both scripts on the same env so kernels behave identically.
image = (
    modal.Image.from_registry(
        "flashinfer/flashinfer-ci-cu132:20260401-2c675fb",
        add_python="3.12",
    )
    .pip_install(
        "cupti-python",
        "nvidia-cutlass-dsl",
    )
)


@app.function(image=image, gpu=MODAL_GPU, timeout=600,
              volumes={TRACE_PATH: trace_volume})
def profile_kernel(track: str, sources: dict, entry_point: str):
    import importlib.util
    import os
    import tempfile
    import torch
    import torch.profiler

    # ── Materialize sources to a temp dir and import the entry module ───────
    entry_file, entry_func = entry_point.split("::")
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, content in sources.items():
            with open(os.path.join(tmpdir, fname), "w") as f:
                f.write(content)
        sys.path.insert(0, tmpdir)
        spec = importlib.util.spec_from_file_location(
            "entry_kernel", os.path.join(tmpdir, entry_file),
        )
        kern_mod = importlib.util.module_from_spec(spec)
        sys.modules["entry_kernel"] = kern_mod
        spec.loader.exec_module(kern_mod)
        kernel_fn = getattr(kern_mod, entry_func)

    # ── Track-specific synthetic inputs ─────────────────────────────────────
    torch.manual_seed(42)
    device = "cuda"

    if track != "moe":
        # Add other tracks here when needed (dsa_indexer / dsa_attention have
        # different signatures; profiling MoE is the immediate need).
        raise NotImplementedError(
            f"profile_kernel currently only supports --track moe (got {track!r})"
        )

    # MoE workload: medium-large representative size.
    T = 2048
    H = 7168
    I_DIM = 2048
    E_LOCAL = 32
    E_GLOBAL = 256
    QK = 128

    def block_quant(x, b0=QK, b1=QK):
        n0 = x.shape[0] // b0
        n1 = x.shape[1] // b1
        xr = x.float().reshape(n0, b0, n1, b1)
        sc = xr.abs().amax(dim=(1, 3)) / 448.0 + 1e-12
        xq = (xr / sc[:, None, :, None]).reshape(x.shape)
        return xq.to(torch.float8_e4m3fn), sc

    h_f32 = torch.randn(T, H, device=device) * 0.1
    h_sc = h_f32.reshape(T, H // QK, QK).abs().amax(dim=2) / 448.0 + 1e-12
    h_fp8 = (h_f32.reshape(T, H // QK, QK) / h_sc.unsqueeze(2)).reshape(T, H).to(torch.float8_e4m3fn)
    hidden_states_scale = h_sc.T.contiguous()

    W1_fp8_list, W1_sc_list = [], []
    W2_fp8_list, W2_sc_list = [], []
    for _ in range(E_LOCAL):
        fp8, sc = block_quant(torch.randn(2 * I_DIM, H, device=device) * 0.1)
        W1_fp8_list.append(fp8); W1_sc_list.append(sc)
        fp8, sc = block_quant(torch.randn(H, I_DIM, device=device) * 0.1)
        W2_fp8_list.append(fp8); W2_sc_list.append(sc)

    gemm1_weights = torch.stack(W1_fp8_list)
    gemm1_weights_scale = torch.stack(W1_sc_list)
    gemm2_weights = torch.stack(W2_fp8_list)
    gemm2_weights_scale = torch.stack(W2_sc_list)

    routing_logits = torch.randn(T, E_GLOBAL, device=device)
    routing_bias = torch.zeros(E_GLOBAL, device=device)
    local_expert_offset = 0
    routed_scaling_factor = 1.0
    output = torch.zeros(T, H, device=device, dtype=torch.bfloat16)

    def run():
        output.zero_()
        kernel_fn(
            routing_logits, routing_bias,
            h_fp8, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor,
            output,
        )
        torch.cuda.synchronize()

    # ── Warmup (also triggers Triton autotuning) ────────────────────────────
    print(f"Warming up '{track}' kernel via {entry_point} ...")
    for _ in range(5):
        run()
    print("Warmup done.")

    # ── Profile ─────────────────────────────────────────────────────────────
    trace_file = f"{TRACE_PATH}/kernel_profile.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(10):
            run()

    prof.export_chrome_trace(trace_file)
    print(f"\nTrace written to {trace_file}")

    print("\n── CUDA kernel summary (top 15 by total CUDA time) ─────────────")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def _load_track(track: str) -> tuple[dict, str]:
    """Local-side: gather <track>/solution/triton/*.py and read entry_point."""
    track_dir = PROJECT_ROOT / track
    config_path = track_dir / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Track config not found: {config_path}")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    language = config["build"]["language"]
    src_dir = track_dir / "solution" / language
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    sources = {f.name: f.read_text() for f in src_dir.glob("*.py")}
    if not sources:
        raise FileNotFoundError(f"No .py files in {src_dir}")
    return sources, config["build"]["entry_point"]


@app.local_entrypoint()
def main(track: str = "moe"):
    sources, entry_point = _load_track(track)
    print(f"Track: {track}")
    print(f"Entry: {entry_point}")
    print(f"Files: {sorted(sources.keys())}")
    print(f"GPU:   {MODAL_GPU}")
    print(f"Profiling on Modal {MODAL_GPU} ...")
    profile_kernel.remote(track, sources, entry_point)
    print("\nDone. Download the trace:")
    print("  modal volume get flashinfer-trace kernel_profile.json")
    print("Then open kernel_profile.json at https://ui.perfetto.dev")
