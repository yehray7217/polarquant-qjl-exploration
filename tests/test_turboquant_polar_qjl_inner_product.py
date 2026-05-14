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
    estimate_stage1_bits_per_coordinate,
)
from turboquant.qjl import (
    make_gaussian_sketch,
    qjl_encode,
    qjl_inner_product_estimate,
)


@torch.no_grad()
def dot_product_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """
    pred, target: [N]
    """
    pred_f = pred.to(torch.float32)
    target_f = target.to(torch.float32)

    err = pred_f - target_f

    abs_err = torch.abs(err)
    sq_err = err * err

    mae = torch.mean(abs_err)
    rmse = torch.sqrt(torch.mean(sq_err))
    bias = torch.mean(err)

    target_abs_mean = torch.mean(torch.abs(target_f))
    relative_mae = mae / torch.clamp(
        target_abs_mean,
        min=1e-12,
    )

    target_rmse = torch.sqrt(
        torch.mean(target_f * target_f)
    )
    relative_rmse = rmse / torch.clamp(
        target_rmse,
        min=1e-12,
    )

    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "bias": float(bias.item()),
        "relative_mae": float(relative_mae.item()),
        "relative_rmse": float(relative_rmse.item()),
        "max_abs_error": float(abs_err.max().item()),
    }


@torch.no_grad()
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # ============================================================
    # Configuration
    # ============================================================

    D = 128
    L = 4
    M = 128

    N_calib = 4096
    N_eval = 4096

    # ============================================================
    # 1. Fit Polar Stage-1 angle codebooks
    #    In the final SVD-LLaMA pipeline, these samples will come
    #    from real calibration K activations.
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
    # 2. Build held-out vectors and random queries
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

    # ============================================================
    # 3. TurboQuant Stage 1:
    #    x -> polar encode -> angle quantize -> decode -> x_hat_polar
    # ============================================================

    enc = recursive_polar_encode(
        x,
        num_levels=L,
    )

    qenc = quantize_polar_encoding(
        encoding=enc,
        codebooks=codebooks,
    )

    deq = dequantize_polar_encoding(
        qencoding=qenc,
        codebooks=codebooks,
    )

    x_hat_polar = recursive_polar_decode(
        deq,
    )

    # ============================================================
    # 4. Residual for TurboQuant Stage 2
    # ============================================================

    residual = x - x_hat_polar

    # ============================================================
    # 5. QJL encode residual
    # ============================================================

    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    residual_qjl = qjl_encode(
        x=residual,
        S=sketch,
    )

    residual_correction = qjl_inner_product_estimate(
        q=q,
        encoded_r=residual_qjl,
        S=sketch,
    )

    # ============================================================
    # 6. Compare inner products
    # ============================================================

    true_dot = torch.sum(
        q * x,
        dim=-1,
    )

    polar_stage1_dot = torch.sum(
        q * x_hat_polar,
        dim=-1,
    )

    turboquant_prod_dot = (
        polar_stage1_dot +
        residual_correction
    )

    stage1_metrics = dot_product_metrics(
        pred=polar_stage1_dot,
        target=true_dot,
    )

    full_metrics = dot_product_metrics(
        pred=turboquant_prod_dot,
        target=true_dot,
    )

    # ============================================================
    # 7. Auxiliary reconstruction metrics
    # ============================================================

    residual_l2 = torch.linalg.vector_norm(
        residual,
        ord=2,
        dim=-1,
    ).mean()

    x_l2 = torch.linalg.vector_norm(
        x,
        ord=2,
        dim=-1,
    ).mean()

    mean_relative_residual_l2 = (
        residual_l2 /
        torch.clamp(x_l2, min=1e-12)
    )

    rmse_improvement = (
        stage1_metrics["rmse"] /
        max(full_metrics["rmse"], 1e-12)
    )

    relative_rmse_improvement = (
        stage1_metrics["relative_rmse"] /
        max(full_metrics["relative_rmse"], 1e-12)
    )

    bits_per_coord_stage1 = estimate_stage1_bits_per_coordinate(
        original_dim=D,
        num_levels=L,
        bits_by_level=DEFAULT_POLAR_BITS_BY_LEVEL,
        radius_bits=16,
    )

    # Conceptual TurboQuant total:
    # Stage 1 PolarQuant 3.875 bits/channel
    # + 1-bit QJL residual sketch is reported separately here,
    # because its amortized accounting depends on sketch dimension M
    # and stored metadata.
    #
    # For this correctness test we only print Stage-1 storage ratio.

    # ============================================================
    # 8. Console report
    # ============================================================

    print("========== TurboQuant Polar Stage-1 + QJL residual inner-product test ==========")
    print(f"device                         = {device}")
    print(f"D                              = {D}")
    print(f"L                              = {L}")
    print(f"M                              = {M}")
    print(f"N_calib                        = {N_calib}")
    print(f"N_eval                         = {N_eval}")
    print(f"bits_by_level                  = {DEFAULT_POLAR_BITS_BY_LEVEL}")
    print(f"Stage-1 bits/channel           = {bits_per_coord_stage1:.6f}")
    print()

    print("----- Stage-1 polar reconstruction -----")
    print(
        f"mean relative residual L2      = "
        f"{float(mean_relative_residual_l2.item()):.6e}"
    )
    print()

    print("----- Inner-product error: Polar Stage 1 only -----")
    print(f"MAE                            = {stage1_metrics['mae']:.6e}")
    print(f"RMSE                           = {stage1_metrics['rmse']:.6e}")
    print(f"bias                           = {stage1_metrics['bias']:.6e}")
    print(f"relative MAE                   = {stage1_metrics['relative_mae']:.6e}")
    print(f"relative RMSE                  = {stage1_metrics['relative_rmse']:.6e}")
    print(f"max_abs_error                  = {stage1_metrics['max_abs_error']:.6e}")
    print()

    print("----- Inner-product error: Polar Stage 1 + QJL residual -----")
    print(f"MAE                            = {full_metrics['mae']:.6e}")
    print(f"RMSE                           = {full_metrics['rmse']:.6e}")
    print(f"bias                           = {full_metrics['bias']:.6e}")
    print(f"relative MAE                   = {full_metrics['relative_mae']:.6e}")
    print(f"relative RMSE                  = {full_metrics['relative_rmse']:.6e}")
    print(f"max_abs_error                  = {full_metrics['max_abs_error']:.6e}")
    print()

    print("----- Improvement -----")
    print(f"RMSE improvement               = {rmse_improvement:.6f}x")
    print(f"relative RMSE improvement      = {relative_rmse_improvement:.6f}x")

    # ============================================================
    # 9. Assertions
    # ============================================================

    assert x_hat_polar.shape == x.shape
    assert true_dot.shape == polar_stage1_dot.shape
    assert true_dot.shape == turboquant_prod_dot.shape

    # The key correctness criterion:
    # QJL residual correction should reduce aggregate inner-product RMSE.
    assert full_metrics["rmse"] < stage1_metrics["rmse"]

    print("[PASS] TurboQuant Polar Stage-1 + QJL residual inner-product test passed.")


if __name__ == "__main__":
    main()
