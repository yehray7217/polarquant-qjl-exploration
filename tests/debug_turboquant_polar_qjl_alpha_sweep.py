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
from turboquant.qjl import (
    QJL_CORRECTION_SCALE,
    make_gaussian_sketch,
    qjl_encode,
    qjl_inner_product_estimate,
)

@torch.no_grad()
def _metrics(
    *,
    pred: torch.Tensor,
    ref: torch.Tensor,
) -> dict[str, float]:
    err = pred.to(torch.float32) - ref.to(torch.float32)

    mae = torch.mean(torch.abs(err))
    rmse = torch.sqrt(torch.mean(err * err))
    bias = torch.mean(err)

    ref_rms = torch.sqrt(
        torch.mean(ref.to(torch.float32) ** 2)
    )

    rel_rmse = rmse / torch.clamp(
        ref_rms,
        min=1e-12,
    )

    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "bias": float(bias.item()),
        "relative_rmse": float(rel_rmse.item()),
    }


@torch.no_grad()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = "cuda:0"

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    D = 128
    L = 4
    M = 128
    N_calib = 4096
    N_eval = 4096

    bits_by_level = DEFAULT_POLAR_BITS_BY_LEVEL

    # ------------------------------------------------------------
    # Calibration vectors for Polar codebooks
    # ------------------------------------------------------------
    x_calib = torch.randn(
        N_calib,
        D,
        device=device,
        dtype=torch.float32,
    )

    enc_calib = recursive_polar_encode(
        x_calib,
        num_levels=L,
    )

    codebooks = fit_polar_angle_codebooks_from_encodings(
        [enc_calib],
        bits_by_level=bits_by_level,
        max_iters=30,
        max_samples_per_level=200_000,
        seed=0,
    )

    # ------------------------------------------------------------
    # Evaluation vectors and queries
    # ------------------------------------------------------------
    x_eval = torch.randn(
        N_eval,
        D,
        device=device,
        dtype=torch.float32,
    )

    q_eval = torch.randn(
        N_eval,
        D,
        device=device,
        dtype=torch.float32,
    )

    dense_ip = torch.sum(
        q_eval * x_eval,
        dim=-1,
    )

    # ------------------------------------------------------------
    # Stage 1 Polar reconstruction
    # ------------------------------------------------------------
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

    x_hat_stage1 = recursive_polar_decode(
        deq_eval,
    ).to(torch.float32)

    stage1_ip = torch.sum(
        q_eval * x_hat_stage1,
        dim=-1,
    )

    residual = (
        x_eval.to(torch.float32)
        - x_hat_stage1.to(torch.float32)
    )

    # ------------------------------------------------------------
    # QJL residual correction
    # ------------------------------------------------------------
    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    qjl_residual = qjl_encode(
        x=residual,
        S=sketch,
    )

    qjl_correction = qjl_inner_product_estimate(
        q=q_eval,
        encoded_r=qjl_residual,
        S=sketch,
    )
    # qjl_inner_product_estimate() already includes the global
    # QJL_CORRECTION_SCALE. Recover the raw estimator so that the
    # sweep below scans the true effective correction scale.
    qjl_correction_raw = (
        qjl_correction / float(QJL_CORRECTION_SCALE)
    )

    # ------------------------------------------------------------
    # Sweep alpha
    # ------------------------------------------------------------
    print("========== QJL alpha sweep for PolarQuant practical (4,2,2,2) + 1-bit QJL ==========")
    print(f"D                    = {D}")
    print(f"L                    = {L}")
    print(f"M                    = {M}")
    print(f"bits_by_level        = {bits_by_level}")
    print()

    stage1_metrics = _metrics(
        pred=stage1_ip,
        ref=dense_ip,
    )

    print("----- Stage 1 only -----")
    print(f"RMSE                 = {stage1_metrics['rmse']:.6e}")
    print(f"relative RMSE        = {stage1_metrics['relative_rmse']:.6e}")
    print(f"bias                 = {stage1_metrics['bias']:.6e}")
    print()

    best_alpha = None
    best_metrics = None

    effective_scales = [
        0.000,
        0.125,
        0.250,
        0.375,
        0.500,
        0.625,
        0.750,
        0.875,
        1.000,
        1.125,
        1.250,
        1.375,
        1.500,
    ]

    print("----- Effective QJL correction scale sweep -----")
    print(f"current global QJL_CORRECTION_SCALE = {QJL_CORRECTION_SCALE:.6f}")
    print(
        f"{'scale':>8} "
        f"{'rmse':>14} "
        f"{'rel_rmse':>14} "
        f"{'bias':>14} "
        f"{'vs_stage1':>14}"
    )

    for effective_scale in effective_scales:
        pred = (
            stage1_ip
            + float(effective_scale) * qjl_correction_raw
        )

        metrics = _metrics(
            pred=pred,
            ref=dense_ip,
        )

        improvement = (
            stage1_metrics["rmse"]
            / metrics["rmse"]
        )

        print(
            f"{effective_scale:8.3f} "
            f"{metrics['rmse']:14.6e} "
            f"{metrics['relative_rmse']:14.6e} "
            f"{metrics['bias']:14.6e} "
            f"{improvement:14.6f}x"
        )

        if (
            best_metrics is None
            or metrics["rmse"] < best_metrics["rmse"]
        ):
            best_alpha = float(effective_scale)
            best_metrics = metrics

    assert best_alpha is not None
    assert best_metrics is not None

    print()
    print("----- Best alpha -----")
    print(f"best effective scale = {best_alpha:.3f}")
    print(f"best RMSE            = {best_metrics['rmse']:.6e}")
    print(f"best relative RMSE   = {best_metrics['relative_rmse']:.6e}")
    print(
        "improvement vs Stage1= "
        f"{stage1_metrics['rmse'] / best_metrics['rmse']:.6f}x"
    )


if __name__ == "__main__":
    main()
