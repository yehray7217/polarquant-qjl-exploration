from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.qjl import (
    make_gaussian_sketch,
    qjl_encode,
    qjl_inner_product_estimate,
)
from turboquant.qjl_score_cuda import (
    qjl_packed_score_cuda,
)


@torch.no_grad()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    device = "cuda:0"

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    B = 1
    H = 32
    Q = 1
    T = 65
    D = 128
    M = 128

    q = torch.randn(
        B,
        H,
        Q,
        D,
        device=device,
        dtype=torch.float32,
    )

    residual = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=torch.float32,
    )

    S = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    residual_flat = residual.reshape(
        B * H * T,
        D,
    )

    enc = qjl_encode(
        x=residual_flat,
        S=S,
    )

    packed_signs = enc.packed_sign_bits.reshape(
        B,
        H,
        T,
        M // 8,
    )

    norms = enc.norms.reshape(
        B,
        H,
        T,
    )

    # ------------------------------------------------------------
    # Reference: current qjl_inner_product_estimate()
    # ------------------------------------------------------------
    q_expanded = q.expand(
        -1,
        -1,
        T,
        -1,
    )

    q_flat = q_expanded.reshape(
        B * H * T,
        D,
    )

    ref_flat = qjl_inner_product_estimate(
        q=q_flat,
        encoded_r=enc,
        S=S,
    )

    scores_ref = ref_flat.reshape(
        B,
        H,
        1,
        T,
    )

    # ------------------------------------------------------------
    # CUDA fused residual score
    # ------------------------------------------------------------
    q_projected = torch.matmul(
        q.to(torch.float32),
        S.T.to(torch.float32),
    )

    scores_cuda = qjl_packed_score_cuda(
        q_projected=q_projected,
        packed_signs=packed_signs,
        norms=norms,
    )

    diff = torch.abs(
        scores_ref.to(torch.float32)
        - scores_cuda.to(torch.float32)
    )

    print("========== Fused QJL packed-sign CUDA score parity ==========")
    print(f"scores_ref.shape   = {tuple(scores_ref.shape)}")
    print(f"scores_cuda.shape  = {tuple(scores_cuda.shape)}")
    print(f"max_abs_diff       = {float(diff.max().item()):.6e}")
    print(f"mean_abs_diff      = {float(diff.mean().item()):.6e}")

    assert tuple(scores_cuda.shape) == (
        B,
        H,
        Q,
        T,
    )

    assert float(diff.max().item()) < 2e-4

    print("[PASS] Fused QJL packed-sign CUDA score parity passed.")


if __name__ == "__main__":
    main()
