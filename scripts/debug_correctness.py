"""
Standalone kernel correctness test on Modal H100.

No flashinfer_bench dependency — creates synthetic FP8 inputs, runs our
kernel, computes a float32 PyTorch reference, and reports errors.

Usage:
    modal run scripts/debug_correctness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app   = modal.App("flashinfer-debug")
image = modal.Image.from_registry(
    "flashinfer/flashinfer-ci-cu132:20260401-2c675fb",
    add_python="3.12",
)


@app.function(image=image, gpu="H100:1", timeout=600)
def test_kernel(kernel_source: str):
    import importlib, os, sys, tempfile
    import torch
    import torch.nn.functional as F

    # ── import our kernel ────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "kernel.py"), "w") as f:
            f.write(kernel_source)
        sys.path.insert(0, tmpdir)
        kern = importlib.import_module("kernel")

    torch.manual_seed(42)
    device = "cuda"

    T        = 2048    # tokens — stress-test with large batch
    H        = 7168
    I_DIM    = 2048
    E_LOCAL  = 32
    E_GLOBAL = 256
    TOP_K    = 8
    QK       = 128     # FP8 quantization block

    # ── helper: block-quantize a 2-D tensor to FP8 ──────────────────────────
    def block_quant(x, axis0_block=QK, axis1_block=QK):
        """x [N, K] → fp8 [N, K], scale [N//b0, K//b1]"""
        n0 = x.shape[0] // axis0_block
        n1 = x.shape[1] // axis1_block
        xr = x.float().reshape(n0, axis0_block, n1, axis1_block)
        scale = xr.abs().amax(dim=(1, 3)) / 448.0 + 1e-12   # [n0, n1]
        xq = (xr / scale[:, None, :, None]).reshape(x.shape)
        return xq.to(torch.float8_e4m3fn), scale

    def block_dequant(fp8, scale, axis0_block=QK, axis1_block=QK):
        """Inverse of block_quant."""
        f32 = fp8.to(torch.float32)
        n0, n1 = scale.shape
        f32 = f32.reshape(n0, axis0_block, n1, axis1_block)
        f32 = f32 * scale[:, None, :, None]
        return f32.reshape(fp8.shape)

    # ── build synthetic inputs ───────────────────────────────────────────────
    # hidden_states [T, H] FP8, scale [H//128, T]
    # Quantize per-row along H only (T=64 < QK=128, so can't block along token dim)
    h_f32 = torch.randn(T, H, device=device) * 0.1
    h_scale_raw = h_f32.reshape(T, H // QK, QK).abs().amax(dim=2) / 448.0 + 1e-12  # [T, H//128]
    h_fp8 = (h_f32.reshape(T, H // QK, QK) / h_scale_raw.unsqueeze(2)).reshape(T, H).to(torch.float8_e4m3fn)
    hidden_states_scale = h_scale_raw.T.contiguous()   # [H//128, T]

    # W1: [E_LOCAL, 2*I, H] FP8, scale [E_LOCAL, 2*I//128, H//128]
    W1_f32 = torch.randn(E_LOCAL, 2 * I_DIM, H, device=device) * 0.1
    W1_fp8_list, W1_sc_list = [], []
    for e in range(E_LOCAL):
        fp8, sc = block_quant(W1_f32[e])
        W1_fp8_list.append(fp8)
        W1_sc_list.append(sc)
    gemm1_weights       = torch.stack(W1_fp8_list)   # [E, 2*I, H]
    gemm1_weights_scale = torch.stack(W1_sc_list)    # [E, 2*I//128, H//128]

    # W2: [E_LOCAL, H, I] FP8, scale [E_LOCAL, H//128, I//128]
    W2_f32 = torch.randn(E_LOCAL, H, I_DIM, device=device) * 0.1
    W2_fp8_list, W2_sc_list = [], []
    for e in range(E_LOCAL):
        fp8, sc = block_quant(W2_f32[e])
        W2_fp8_list.append(fp8)
        W2_sc_list.append(sc)
    gemm2_weights       = torch.stack(W2_fp8_list)   # [E, H, I]
    gemm2_weights_scale = torch.stack(W2_sc_list)    # [E, H//128, I//128]

    routing_logits        = torch.randn(T, E_GLOBAL, device=device)
    routing_bias          = torch.zeros(E_GLOBAL, device=device)
    local_expert_offset   = 0
    routed_scaling_factor = 1.0

    # ── run our kernel ───────────────────────────────────────────────────────
    output = torch.zeros(T, H, device=device, dtype=torch.bfloat16)
    kern.kernel(
        routing_logits, routing_bias,
        h_fp8, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor,
        output,
    )
    out_f32 = output.float()

    # ── reference: pure float32 PyTorch ─────────────────────────────────────
    # Dequantize hidden states
    h_dq = h_fp8.to(torch.float32) * hidden_states_scale.T.repeat_interleave(QK, dim=1)  # [T, H]

    # Reference routing (mirrors _routing_kernel logic)
    s_f32   = routing_logits.float().sigmoid()           # [T, 256]
    s_bias  = s_f32 + routing_bias.float()
    top2    = s_bias.reshape(T, 8, 32).topk(2, dim=2).values.sum(dim=2)  # [T, 8]
    top4_g  = top2.topk(4, dim=1).indices                # [T, 4]
    egate   = torch.zeros(T, 256, device=device, dtype=torch.bool)
    for gi in range(4):
        starts = (top4_g[:, gi] * 32).unsqueeze(1)       # [T, 1]
        egate.scatter_(1, starts + torch.arange(32, device=device), True)
    masked       = s_bias.masked_fill(~egate, float('-inf'))
    topk_idx_ref = masked.topk(8, dim=1).indices          # [T, 8]
    topk_s_ref   = s_f32.gather(1, topk_idx_ref)
    topk_w_ref   = topk_s_ref / (topk_s_ref.sum(1, keepdim=True) + 1e-20) * routed_scaling_factor

    ref = torch.zeros(T, H, device=device, dtype=torch.float32)
    for e_local in range(E_LOCAL):
        e_global = e_local  # local_expert_offset = 0
        e_mask   = (topk_idx_ref == e_global)             # [T, 8]
        tok_ids  = e_mask.any(dim=1).nonzero(as_tuple=True)[0]
        if tok_ids.numel() == 0:
            continue
        tok_w = (topk_w_ref[tok_ids] * e_mask[tok_ids].float()).sum(dim=1)  # [Tk]

        W1_dq = block_dequant(W1_fp8_list[e_local], W1_sc_list[e_local])
        g1    = h_dq[tok_ids] @ W1_dq.T                      # [Tk, 2*I]
        swiglu = F.silu(g1[:, I_DIM:]) * g1[:, :I_DIM]       # [Tk, I]

        W2_dq = block_dequant(W2_fp8_list[e_local], W2_sc_list[e_local])
        g2    = swiglu @ W2_dq.T                              # [Tk, H]
        ref[tok_ids] += tok_w.unsqueeze(1) * g2

    # ── compare ──────────────────────────────────────────────────────────────
    diff    = (out_f32 - ref).abs()
    rel_err = diff / (ref.abs() + 1e-6)

    max_abs = diff.max().item()
    max_rel = rel_err.max().item()
    mean_abs = diff.mean().item()
    pass_ratio = ((diff < 1.0 + 0.3 * ref.abs()).float().mean()).item()

    worst = diff.argmax()
    wr, wc = worst.item() // H, worst.item() % H

    print(f"Max  abs error : {max_abs:.4e}")
    print(f"Max  rel error : {max_rel:.4e}")
    print(f"Mean abs error : {mean_abs:.4e}")
    print(f"Pass ratio (atol=1,rtol=0.3): {pass_ratio:.1%}")
    print(f"Worst element  : ours={out_f32[wr,wc]:.4f}  ref={ref[wr,wc]:.4f}  (row={wr}, col={wc})")

    return {"max_abs": max_abs, "max_rel": max_rel, "pass_ratio": pass_ratio}


@app.local_entrypoint()
def main():
    kernel_source = (PROJECT_ROOT / "moe" / "solution" / "triton" / "kernel.py").read_text()
    print("Uploading kernel and running correctness test on H100...")
    result = test_kernel.remote(kernel_source)
    print(f"\nFinal: {result}")
