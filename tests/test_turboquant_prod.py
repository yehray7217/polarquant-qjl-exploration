from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.mse_quant import (
    make_random_rotation,
    get_2bit_centroids,
)
from turboquant.qjl import make_gaussian_sketch
from turboquant.prod_quant import (
    turboquant_prod_quantize_3bit,
    turboquant_prod_mse_reconstruction,
    turboquant_prod_inner_product_estimate,
)


@torch.no_grad()
def main():
    torch.manual_seed(0)

    d = 128
    n_trials = 8192

    # 3-bit TurboQuant_prod prototype:
    # - 2-bit MSE stage
    # - 1-bit QJL residual
    #
    # QJL sketch dimension: start with 256.
    qjl_m = 256

    rotation = make_random_rotation(
        d=d,
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )

    centroids = get_2bit_centroids(
        d=d,
        device="cpu",
        dtype=torch.float32,
    )

    sketch = make_gaussian_sketch(
        d=d,
        m=qjl_m,
        device="cpu",
        dtype=torch.float32,
        seed=456,
    )

    # q = query-like vector
    # x = key-like vector
    q = torch.randn(n_trials, d)
    x = torch.randn(n_trials, d)

    enc = turboquant_prod_quantize_3bit(
        x=x,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    x_hat_mse = turboquant_prod_mse_reconstruction(
        encoding=enc,
        rotation=rotation,
        centroids=centroids,
    )

    true_dot = torch.sum(q * x, dim=-1)
    mse_dot = torch.sum(q * x_hat_mse, dim=-1)
    prod_dot = turboquant_prod_inner_product_estimate(
        q=q,
        encoding=enc,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    mse_err = mse_dot - true_dot
    prod_err = prod_dot - true_dot

    mse_bias = torch.mean(mse_err).item()
    prod_bias = torch.mean(prod_err).item()

    mse_mae = torch.mean(torch.abs(mse_err)).item()
    prod_mae = torch.mean(torch.abs(prod_err)).item()

    mse_rmse = torch.sqrt(torch.mean(mse_err ** 2)).item()
    prod_rmse = torch.sqrt(torch.mean(prod_err ** 2)).item()

    mean_true_mag = torch.mean(torch.abs(true_dot)).item()
    mse_norm_mae = mse_mae / max(mean_true_mag, 1e-12)
    prod_norm_mae = prod_mae / max(mean_true_mag, 1e-12)

    mse_corr = torch.corrcoef(torch.stack([mse_dot, true_dot]))[0, 1].item()
    prod_corr = torch.corrcoef(torch.stack([prod_dot, true_dot]))[0, 1].item()

    print("========== TurboQuant_prod 3-bit test ==========")
    print(f"d                         = {d}")
    print(f"n_trials                  = {n_trials}")
    print(f"QJL sketch m              = {qjl_m}")
    print()
    print("----- MSE-only inner product -----")
    print(f"bias                      = {mse_bias:.6e}")
    print(f"MAE                       = {mse_mae:.6e}")
    print(f"RMSE                      = {mse_rmse:.6e}")
    print(f"normalized MAE            = {mse_norm_mae:.6f}")
    print(f"corr(est, true)           = {mse_corr:.6f}")
    print()
    print("----- TurboQuant_prod inner product -----")
    print(f"bias                      = {prod_bias:.6e}")
    print(f"MAE                       = {prod_mae:.6e}")
    print(f"RMSE                      = {prod_rmse:.6e}")
    print(f"normalized MAE            = {prod_norm_mae:.6f}")
    print(f"corr(est, true)           = {prod_corr:.6f}")
    print()

    # What we want to see:
    # 1. prod bias should be closer to zero than MSE-only bias in expectation.
    # 2. prod should not significantly worsen correlation.
    #
    # The exact finite-sample values fluctuate, so keep thresholds modest.
    assert abs(prod_bias) < 0.25, f"prod bias too large: {prod_bias}"
    assert prod_corr > 0.85, f"prod correlation too low: {prod_corr}"

    print("[PASS] TurboQuant_prod 3-bit sanity check passed.")


if __name__ == "__main__":
    main()
