from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.polarquant import (
    recursive_polar_encode,
)
from turboquant.polarquant_quant import (
    DEFAULT_POLAR_BITS_BY_LEVEL,
    fit_polar_angle_codebooks_from_encodings,
)
from turboquant.polar_prod import (
    turboquant_polar_prod_quantize,
    turboquant_polar_prod_reconstruction,
    turboquant_polar_prod_inner_product_estimate,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)


@torch.no_grad()
def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    err = pred.to(torch.float32) - target.to(torch.float32)
    return float(torch.sqrt(torch.mean(err * err)).item())


@torch.no_grad()
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    D = 128
    L = 4
    M = 128

    N_calib = 4096
    N_eval = 4096

    # ============================================================
    # Calibration codebooks
    # ============================================================

    x_calib = torch.randn(
        N_calib,
        D,
        device=device,
        dtype=dtype,
    )

    enc_calib = recursive_polar_encode(
        x_calib,
        num_levels=L,
    )

    codebooks = fit_polar_angle_codebooks_from_encodings(
        [enc_calib],
        bits_by_level=DEFAULT_POLAR_BITS_BY_LEVEL,
        max_iters=30,
        max_samples_per_level=200_000,
        seed=0,
    )

    # ============================================================
    # Eval tensors
    # ============================================================

    x = torch.randn(
        N_eval,
        D,
        device=device,
        dtype=dtype,
    )

    q = torch.randn(
        N_eval,
        D,
        device=device,
        dtype=dtype,
    )

    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    # ============================================================
    # Polar-prod module API
    # ============================================================

    encoding = turboquant_polar_prod_quantize(
        x=x,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=L,
    )

    x_hat_polar = turboquant_polar_prod_reconstruction(
        encoding=encoding,
        codebooks=codebooks,
    )

    est_prod = turboquant_polar_prod_inner_product_estimate(
        q=q,
        encoding=encoding,
        codebooks=codebooks,
        sketch=sketch,
    )

    true_dot = torch.sum(
        q * x,
        dim=-1,
    )

    stage1_dot = torch.sum(
        q * x_hat_polar,
        dim=-1,
    )

    rmse_stage1 = rmse(
        pred=stage1_dot,
        target=true_dot,
    )

    rmse_prod = rmse(
        pred=est_prod,
        target=true_dot,
    )

    improvement = rmse_stage1 / max(rmse_prod, 1e-12)

    print("========== TurboQuant polar-prod module test ==========")
    print(f"device                    = {device}")
    print(f"D                         = {D}")
    print(f"L                         = {L}")
    print(f"M                         = {M}")
    print(f"RMSE Stage 1 only         = {rmse_stage1:.6e}")
    print(f"RMSE Polar + QJL prod     = {rmse_prod:.6e}")
    print(f"RMSE improvement          = {improvement:.6f}x")

    assert x_hat_polar.shape == x.shape
    assert est_prod.shape == true_dot.shape
    assert rmse_prod < rmse_stage1

    print("[PASS] TurboQuant polar-prod module test passed.")


if __name__ == "__main__":
    main()
