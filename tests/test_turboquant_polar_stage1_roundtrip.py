from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from turboquant.polarquant import (
    recursive_polar_encode,
    recursive_polar_decode,
)
from turboquant.polarquant_quant import (
    DEFAULT_POLAR_BITS_BY_LEVEL,
    fit_polar_angle_codebooks_from_encodings,
    quantize_polar_encoding,
    dequantize_polar_encoding,
    estimate_stage1_bits_per_coordinate,
)


@torch.no_grad()
def reconstruction_metrics(
    x: torch.Tensor,
    x_hat: torch.Tensor,
) -> dict[str, float]:
    diff = x_hat.to(torch.float32) - x.to(torch.float32)

    mse = torch.mean(diff * diff)

    x_norm = torch.linalg.vector_norm(
        x.to(torch.float32),
        ord=2,
        dim=-1,
    )

    diff_norm = torch.linalg.vector_norm(
        diff,
        ord=2,
        dim=-1,
    )

    relative_l2 = torch.mean(
        diff_norm / torch.clamp(x_norm, min=1e-12)
    )

    cosine = F.cosine_similarity(
        x.to(torch.float32),
        x_hat.to(torch.float32),
        dim=-1,
    ).mean()

    max_abs_diff = torch.max(torch.abs(diff))
    mean_abs_diff = torch.mean(torch.abs(diff))

    return {
        "mse": float(mse.item()),
        "relative_l2": float(relative_l2.item()),
        "mean_cosine": float(cosine.item()),
        "max_abs_diff": float(max_abs_diff.item()),
        "mean_abs_diff": float(mean_abs_diff.item()),
    }


@torch.no_grad()
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    torch.manual_seed(0)

    B_train = 512
    B_eval = 256
    D = 128
    L = 4

    # ============================================================
    # Calibration-like vectors for fitting per-level angle codebooks
    # ============================================================

    x_train = torch.randn(
        B_train,
        D,
        device=device,
        dtype=dtype,
    )

    enc_train = recursive_polar_encode(
        x_train,
        num_levels=L,
    )

    codebooks = fit_polar_angle_codebooks_from_encodings(
        [enc_train],
        bits_by_level=DEFAULT_POLAR_BITS_BY_LEVEL,
        max_iters=30,
        max_samples_per_level=200_000,
        seed=0,
    )

    # ============================================================
    # Held-out evaluation vectors
    # ============================================================

    x_eval = torch.randn(
        B_eval,
        D,
        device=device,
        dtype=dtype,
    )

    enc_eval = recursive_polar_encode(
        x_eval,
        num_levels=L,
    )

    qenc_eval = quantize_polar_encoding(
        encoding=enc_eval,
        codebooks=codebooks,
    )

    deq_eval = dequantize_polar_encoding(
        qencoding=qenc_eval,
        codebooks=codebooks,
    )

    x_hat = recursive_polar_decode(
        deq_eval,
    )

    metrics = reconstruction_metrics(
        x_eval,
        x_hat,
    )

    bits_per_coord = estimate_stage1_bits_per_coordinate(
        original_dim=D,
        num_levels=L,
        bits_by_level=DEFAULT_POLAR_BITS_BY_LEVEL,
        radius_bits=16,
    )

    # ============================================================
    # Console report
    # ============================================================

    print("========== TurboQuant Stage-1 polar angle quantization ==========")
    print(f"device                    = {device}")
    print(f"D                         = {D}")
    print(f"num_levels                = {L}")
    print(f"bits_by_level             = {DEFAULT_POLAR_BITS_BY_LEVEL}")
    print(f"estimated bits/channel    = {bits_per_coord:.6f}")
    print()

    print("----- Shapes -----")
    print(f"x_eval.shape              = {tuple(x_eval.shape)}")
    print(f"x_hat.shape               = {tuple(x_hat.shape)}")
    print(f"remaining radii shape     = {tuple(enc_eval.radii.shape)}")
    print(
        "angle shapes              = "
        + str([tuple(a.shape) for a in enc_eval.angles])
    )
    print()

    print("----- Codebooks -----")
    for level_idx, centroids in enumerate(
        codebooks.centroids,
        start=1,
    ):
        print(
            f"level {level_idx}: "
            f"num_centroids={centroids.numel()}, "
            f"min={float(centroids.min().item()):.6f}, "
            f"max={float(centroids.max().item()):.6f}"
        )
    print()

    print("----- Reconstruction metrics -----")
    print(f"MSE                       = {metrics['mse']:.6e}")
    print(f"relative L2               = {metrics['relative_l2']:.6e}")
    print(f"mean cosine similarity    = {metrics['mean_cosine']:.6f}")
    print(f"max_abs_diff              = {metrics['max_abs_diff']:.6e}")
    print(f"mean_abs_diff             = {metrics['mean_abs_diff']:.6e}")

    # We do not set an extremely tight quality threshold yet;
    # this test is validating Stage-1 quantization plumbing.
    assert x_hat.shape == x_eval.shape
    assert bits_per_coord == 3.875
    assert metrics["mean_cosine"] > 0.80

    print("[PASS] TurboQuant Stage-1 polar angle quantization passed.")


if __name__ == "__main__":
    main()
