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
from turboquant.qjl_packing import (
    unpack_qjl_signs_1bit,
)


@torch.no_grad()
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    N = 4096
    D = 128
    M = 256

    q = torch.randn(
        N,
        D,
        device=device,
        dtype=torch.float32,
    )

    r = torch.randn(
        N,
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

    enc = qjl_encode(
        x=r,
        S=S,
    )

    # ============================================================
    # Current packed estimator
    # ============================================================

    packed_est = qjl_inner_product_estimate(
        q=q,
        encoded_r=enc,
        S=S,
    )

    # ============================================================
    # Inspect unpacked stored signs
    # ============================================================

    unpacked_signs = unpack_qjl_signs_1bit(
        enc.packed_sign_bits
    )

    # ============================================================
    # Reconstruct the "old dense-sign" estimator directly.
    #
    # IMPORTANT:
    # This mirrors the usual QJL implementation shape:
    #   projected_q = q @ S.T
    #   estimate = residual_norm * <projected_q, signs> / sqrt(M)
    #
    # If your qjl.py uses an additional scaling constant,
    # this diagnostic will reveal a scale mismatch.
    # ============================================================

    q_proj = q @ S.T

    dense_like_est_sqrt_m = (
        enc.norms *
        torch.sum(
            q_proj * unpacked_signs,
            dim=-1,
        )
        / (float(M) ** 0.5)
    )

    dense_like_est_m = (
        enc.norms *
        torch.sum(
            q_proj * unpacked_signs,
            dim=-1,
        )
        / float(M)
    )

    # ============================================================
    # Compare
    # ============================================================

    diff_sqrt_m = torch.abs(
        packed_est - dense_like_est_sqrt_m
    )

    diff_m = torch.abs(
        packed_est - dense_like_est_m
    )

    print("========== QJL packed estimator semantics debug ==========")
    print(f"packed_est.shape              = {tuple(packed_est.shape)}")
    print(f"enc.norms.shape               = {tuple(enc.norms.shape)}")
    print(f"packed_sign_bits.shape        = {tuple(enc.packed_sign_bits.shape)}")
    print(f"unpacked_signs.shape          = {tuple(unpacked_signs.shape)}")
    print()

    print("----- Sign sanity -----")
    print(f"unpacked_signs.min            = {float(unpacked_signs.min().item()):.1f}")
    print(f"unpacked_signs.max            = {float(unpacked_signs.max().item()):.1f}")
    print(f"unpacked_signs.mean           = {float(unpacked_signs.mean().item()):.6e}")
    print()

    print("----- Compare qjl_inner_product_estimate vs manual variants -----")
    print(
        "manual / sqrt(M): "
        f"max_diff={float(diff_sqrt_m.max().item()):.6e}, "
        f"mean_diff={float(diff_sqrt_m.mean().item()):.6e}"
    )
    print(
        "manual / M:       "
        f"max_diff={float(diff_m.max().item()):.6e}, "
        f"mean_diff={float(diff_m.mean().item()):.6e}"
    )
    print()

    # Helpful scale check
    denom = torch.clamp(
        torch.mean(torch.abs(packed_est)),
        min=1e-12,
    )

    print("----- Relative mismatch -----")
    print(
        "sqrt(M) relative mean diff = "
        f"{float(diff_sqrt_m.mean().item() / denom.item()):.6e}"
    )
    print(
        "M relative mean diff       = "
        f"{float(diff_m.mean().item() / denom.item()):.6e}"
    )


if __name__ == "__main__":
    main()
