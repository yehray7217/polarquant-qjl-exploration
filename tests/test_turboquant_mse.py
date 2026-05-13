from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.mse_quant import (
    make_random_rotation,
    get_2bit_centroids,
    turboquant_mse_quantize_2bit,
    turboquant_mse_dequantize_2bit,
)


@torch.no_grad()
def main():
    torch.manual_seed(0)

    d = 128
    n_trials = 4096

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

    # Random vectors; implementation internally normalizes and stores norms.
    x = torch.randn(n_trials, d)

    enc = turboquant_mse_quantize_2bit(
        x=x,
        rotation=rotation,
        centroids=centroids,
    )

    x_hat = turboquant_mse_dequantize_2bit(
        encoding=enc,
        rotation=rotation,
        centroids=centroids,
    )

    residual = x - x_hat

    sq_err = torch.sum(residual ** 2, dim=-1)
    sq_norm = torch.sum(x ** 2, dim=-1)
    relative_mse = torch.mean(sq_err / torch.clamp(sq_norm, min=1e-12)).item()

    residual_norm_ratio = torch.mean(
        torch.linalg.vector_norm(residual, dim=-1)
        / torch.clamp(torch.linalg.vector_norm(x, dim=-1), min=1e-12)
    ).item()

    cosine = torch.nn.functional.cosine_similarity(
        x,
        x_hat,
        dim=-1,
    ).mean().item()

    print("========== TurboQuant_mse 2-bit test ==========")
    print(f"d                       = {d}")
    print(f"n_trials                = {n_trials}")
    print(f"centroids               = {centroids.tolist()}")
    print(f"relative MSE            = {relative_mse:.6f}")
    print(f"residual norm ratio     = {residual_norm_ratio:.6f}")
    print(f"mean cosine similarity  = {cosine:.6f}")

    # Paper reports about 0.117 MSE for b=2 in the asymptotic/high-d setting.
    # d=128 + approximate centroids should be in the same ballpark,
    # so we use forgiving sanity thresholds here.
    assert relative_mse < 0.20, f"relative MSE too high: {relative_mse}"
    assert cosine > 0.90, f"cosine similarity too low: {cosine}"

    print("[PASS] TurboQuant_mse 2-bit sanity check passed.")


if __name__ == "__main__":
    main()
