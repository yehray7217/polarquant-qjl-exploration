from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.polarquant import (
    recursive_polar_encode,
    recursive_polar_decode,
)
from turboquant.polarquant_quant import (
    DEFAULT_POLAR_BITS_BY_LEVEL,
    fit_polar_angle_codebooks_from_encodings,
    quantize_polar_encoding,
    dequantize_polar_encoding,
)
from turboquant.polar_reconstruct_cuda import (
    polar_stage1_reconstruct_cuda,
)


@torch.no_grad()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this test."
        )

    device = "cuda:0"
    dtype = torch.float32

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    B = 1
    H = 32
    T = 65
    D = 128
    L = 4

    N_calib = 4096

    # ------------------------------------------------------------
    # Fit codebooks
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Build sample K tensor
    # ------------------------------------------------------------
    k = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=dtype,
    )

    enc_k = recursive_polar_encode(
        k,
        num_levels=L,
    )

    qenc_k = quantize_polar_encoding(
        encoding=enc_k,
        codebooks=codebooks,
    )

    # ------------------------------------------------------------
    # Reference reconstruction
    # ------------------------------------------------------------
    deq_k = dequantize_polar_encoding(
        qencoding=qenc_k,
        codebooks=codebooks,
    )

    x_hat_ref = recursive_polar_decode(
        deq_k,
    )

    # ------------------------------------------------------------
    # Fused CUDA reconstruction
    # ------------------------------------------------------------
    packed = qenc_k.packed_angles

    x_hat_cuda = polar_stage1_reconstruct_cuda(
        packed_l1=packed.level1_4bit,
        packed_l2=packed.level2_2bit,
        packed_l3=packed.level3_2bit,
        packed_l4=packed.level4_2bit,
        radii=qenc_k.radii,
        centroids_l1=codebooks.centroids[0],
        centroids_l2=codebooks.centroids[1],
        centroids_l3=codebooks.centroids[2],
        centroids_l4=codebooks.centroids[3],
    )

    diff = torch.abs(
        x_hat_ref.to(torch.float32)
        - x_hat_cuda.to(torch.float32)
    )

    print("========== Fused Polar Stage-1 CUDA reconstruction parity ==========")
    print(f"x_hat_ref.shape   = {tuple(x_hat_ref.shape)}")
    print(f"x_hat_cuda.shape  = {tuple(x_hat_cuda.shape)}")
    print(f"max_abs_diff      = {float(diff.max().item()):.6e}")
    print(f"mean_abs_diff     = {float(diff.mean().item()):.6e}")

    assert tuple(x_hat_cuda.shape) == (
        B,
        H,
        T,
        D,
    )

    assert float(diff.max().item()) < 2e-5

    print(
        "[PASS] Fused Polar Stage-1 CUDA reconstruction parity passed."
    )


if __name__ == "__main__":
    main()
