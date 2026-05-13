from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.qjl import (
    make_gaussian_sketch,
    qjl_encode,
    qjl_inner_product_estimate,
)


@torch.no_grad()
def main():
    torch.manual_seed(0)

    d = 128
    n_trials = 4096

    # QJL sketch dimension.
    # 先從 256 開始，之後可掃 64 / 128 / 256 / 512。
    m = 1024

    S = make_gaussian_sketch(
        d=d,
        m=m,
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )

    q = torch.randn(n_trials, d)
    r = torch.randn(n_trials, d)

    encoded_r = qjl_encode(r, S)
    est = qjl_inner_product_estimate(q, encoded_r, S)

    true_dot = torch.sum(q * r, dim=-1)

    abs_err = torch.abs(est - true_dot)
    rel_err = abs_err / torch.clamp(torch.abs(true_dot), min=1e-6)

    bias = torch.mean(est - true_dot).item()
    mae = torch.mean(abs_err).item()
    rmse = torch.sqrt(torch.mean((est - true_dot) ** 2)).item()
    mean_true_mag = torch.mean(torch.abs(true_dot)).item()
    normalized_mae = mae / max(mean_true_mag, 1e-12)

    corr = torch.corrcoef(torch.stack([est, true_dot]))[0, 1].item()

    print("========== QJL residual estimator test ==========")
    print(f"d                  = {d}")
    print(f"m                  = {m}")
    print(f"n_trials           = {n_trials}")
    print(f"bias               = {bias:.6e}")
    print(f"MAE                = {mae:.6e}")
    print(f"RMSE               = {rmse:.6e}")
    print(f"mean|true_dot|     = {mean_true_mag:.6e}")
    print(f"normalized MAE     = {normalized_mae:.6f}")
    print(f"mean relative err  = {rel_err.mean().item():.6f}")
    print(f"corr(est, true)    = {corr:.6f}")

    # Acceptance checks for correctness sanity.
    # bias 不應該系統性偏大；相關性應明顯為正。
    assert abs(bias) < 0.5, f"QJL estimator bias seems too large: {bias}"
    expected_corr_floor = {
        64: 0.40,
        128: 0.55,
        256: 0.65,
        512: 0.75,
        1024: 0.85,
    }
    assert corr > expected_corr_floor[m], (
        f"QJL estimator correlation too low for m={m}: {corr}"
    )

    print("[PASS] QJL residual estimator sanity check passed.")


if __name__ == "__main__":
    main()
