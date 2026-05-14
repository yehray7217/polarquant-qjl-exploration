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
from turboquant.polar_score_cuda import (
    polar_stage1_score_cuda,
)


@torch.no_grad()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    device = "cuda:0"
    dtype = torch.float32

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    B = 1
    H = 32
    Q = 1
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
    # Build key states and quantized Polar encoding
    # ------------------------------------------------------------
    k = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=dtype,
    )

    q = torch.randn(
        B,
        H,
        Q,
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
    # Python reference reconstruction
    # ------------------------------------------------------------
    deq_k = dequantize_polar_encoding(
        qencoding=qenc_k,
        codebooks=codebooks,
    )

    k_hat = recursive_polar_decode(
        deq_k,
    )

    scores_ref = torch.einsum(
        "bhqd,bhkd->bhqk",
        q.to(torch.float32),
        k_hat.to(torch.float32),
    )

    # ------------------------------------------------------------
    # Fused CUDA Stage-1 score
    # ------------------------------------------------------------
    packed = qenc_k.packed_angles

    scores_cuda = polar_stage1_score_cuda(
        q=q,
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
        scores_ref.to(torch.float32)
        - scores_cuda.to(torch.float32)
    )

    print("========== Fused Polar Stage-1 CUDA score parity ==========")
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

    assert float(diff.max().item()) < 2e-3

    print("[PASS] Fused Polar Stage-1 CUDA score parity passed.")


if __name__ == "__main__":
    main()
