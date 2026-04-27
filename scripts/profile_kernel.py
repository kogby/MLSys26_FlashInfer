"""
PyTorch profiler for the MoE kernel on Modal H100/H200.

Runs the kernel with torch.profiler and writes a Chrome trace JSON to the
flashinfer-trace Modal volume.  Download and open in https://ui.perfetto.dev
or chrome://tracing to see per-kernel timelines.

Usage:
    modal run scripts/profile_kernel.py

After the run:
    modal volume get flashinfer-trace kernel_profile.json
    # drag kernel_profile.json into https://ui.perfetto.dev
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app   = modal.App("flashinfer-profile")
image = modal.Image.from_registry(
    "flashinfer/flashinfer-ci-cu132:20260401-2c675fb",
    add_python="3.12",
)

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_PATH   = "/data"


@app.function(image=image, gpu="H100:1", timeout=600,
              volumes={TRACE_PATH: trace_volume})
def profile_kernel(kernel_source: str):
    import importlib, os, sys, tempfile
    import torch
    import torch.profiler

    # ── import kernel ────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "kernel.py"), "w") as f:
            f.write(kernel_source)
        sys.path.insert(0, tmpdir)
        kern = importlib.import_module("kernel")

    torch.manual_seed(42)
    device = "cuda"

    # ── workload sizes (medium-large: representative of real traffic) ────────
    T        = 2048
    H        = 7168
    I_DIM    = 2048
    E_LOCAL  = 32
    E_GLOBAL = 256
    TOP_K    = 8
    QK       = 128

    def block_quant(x, b0=QK, b1=QK):
        n0 = x.shape[0] // b0
        n1 = x.shape[1] // b1
        xr = x.float().reshape(n0, b0, n1, b1)
        sc = xr.abs().amax(dim=(1, 3)) / 448.0 + 1e-12
        xq = (xr / sc[:, None, :, None]).reshape(x.shape)
        return xq.to(torch.float8_e4m3fn), sc

    # hidden states [T, H]
    h_f32 = torch.randn(T, H, device=device) * 0.1
    h_sc  = h_f32.reshape(T, H//QK, QK).abs().amax(dim=2) / 448.0 + 1e-12
    h_fp8 = (h_f32.reshape(T, H//QK, QK) / h_sc.unsqueeze(2)).reshape(T, H).to(torch.float8_e4m3fn)
    hidden_states_scale = h_sc.T.contiguous()

    # W1 [E, 2I, H], W2 [E, H, I]
    W1_fp8_list, W1_sc_list = [], []
    W2_fp8_list, W2_sc_list = [], []
    for e in range(E_LOCAL):
        fp8, sc = block_quant(torch.randn(2*I_DIM, H, device=device)*0.1)
        W1_fp8_list.append(fp8); W1_sc_list.append(sc)
        fp8, sc = block_quant(torch.randn(H, I_DIM, device=device)*0.1)
        W2_fp8_list.append(fp8); W2_sc_list.append(sc)

    gemm1_weights       = torch.stack(W1_fp8_list)
    gemm1_weights_scale = torch.stack(W1_sc_list)
    gemm2_weights       = torch.stack(W2_fp8_list)
    gemm2_weights_scale = torch.stack(W2_sc_list)

    routing_logits        = torch.randn(T, E_GLOBAL, device=device)
    routing_bias          = torch.zeros(E_GLOBAL, device=device)
    local_expert_offset   = 0
    routed_scaling_factor = 1.0
    output = torch.zeros(T, H, device=device, dtype=torch.bfloat16)

    def run():
        output.zero_()
        kern.kernel(
            routing_logits, routing_bias,
            h_fp8, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor,
            output,
        )
        torch.cuda.synchronize()

    # ── warmup (also triggers Triton autotuning) ─────────────────────────────
    print("Warming up (autotuning Triton kernels) ...")
    for _ in range(5):
        run()
    print("Warmup done.")

    # ── profile ──────────────────────────────────────────────────────────────
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

    # ── summary table (printed to stdout) ────────────────────────────────────
    print("\n── CUDA kernel summary (top 15 by total CUDA time) ──────────────────")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
    ))


@app.local_entrypoint()
def main():
    kernel_source = (PROJECT_ROOT / "solution" / "triton" / "kernel.py").read_text()
    print("Uploading kernel and profiling on H100 ...")
    profile_kernel.remote(kernel_source)
    print("\nDone.  To download the trace:")
    print("  modal volume get flashinfer-trace kernel_profile.json")
    print("Then open kernel_profile.json at https://ui.perfetto.dev")
