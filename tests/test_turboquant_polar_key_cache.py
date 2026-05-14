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
    turboquant_polar_prod_inner_product_estimate,
)
from turboquant.polar_key_cache import (
    PolarProdKeyCache,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)


@torch.no_grad()
def error_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    pred_f = pred.to(torch.float32)
    target_f = target.to(torch.float32)

    err = pred_f - target_f
    abs_err = torch.abs(err)

    mae = torch.mean(abs_err)
    rmse = torch.sqrt(torch.mean(err * err))
    max_abs = torch.max(abs_err)

    denom = torch.sqrt(torch.mean(target_f * target_f))
    relative_rmse = rmse / torch.clamp(
        denom,
        min=1e-12,
    )

    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "relative_rmse": float(relative_rmse.item()),
        "max_abs_diff": float(max_abs.item()),
    }


@torch.no_grad()
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # ============================================================
    # Config
    # ============================================================

    B = 1
    H = 32
    D = 128
    L = 4
    M = 128

    T_prefill = 64
    T_decode_append = 1
    T_total = T_prefill + T_decode_append

    N_calib = 4096

    # ============================================================
    # 1. Fit Stage-1 Polar angle codebooks
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
    # 2. Build sketch
    # ============================================================

    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    # ============================================================
    # 3. Create dense K chunks
    # ============================================================

    k_prefill = torch.randn(
        B,
        H,
        T_prefill,
        D,
        device=device,
        dtype=dtype,
    )

    k_decode = torch.randn(
        B,
        H,
        T_decode_append,
        D,
        device=device,
        dtype=dtype,
    )

    k_dense_all = torch.cat(
        [k_prefill, k_decode],
        dim=2,
    )

    q = torch.randn(
        B,
        H,
        1,
        D,
        device=device,
        dtype=dtype,
    )

    # ============================================================
    # 4. PolarProdKeyCache append
    # ============================================================

    cache = PolarProdKeyCache(
        num_layers=1,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=L,
    )

    cache.append(
        layer_idx=0,
        key_states=k_prefill,
    )

    print("========== After prefill append ==========")
    print(f"seq_len: {cache.seq_len(layer_idx=0)}")
    print(cache.report())
    print()

    assert cache.seq_len(layer_idx=0) == T_prefill

    cache.append(
        layer_idx=0,
        key_states=k_decode,
    )

    print("========== After decode-token append ==========")
    print(f"seq_len: {cache.seq_len(layer_idx=0)}")
    print(cache.report())
    print()

    assert cache.seq_len(layer_idx=0) == T_total

    # ============================================================
    # 5. Cache score
    # ============================================================

    scores_cache = cache.score(
        layer_idx=0,
        query_states=q,
    )

    # Expected shape: [B,H,1,T_total]
    assert tuple(scores_cache.shape) == (
        B,
        H,
        1,
        T_total,
    )

    # ============================================================
    # 6. Direct chunkwise PolarProd reference
    # ============================================================

    enc_prefill = turboquant_polar_prod_quantize(
        x=k_prefill,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=L,
    )

    enc_decode = turboquant_polar_prod_quantize(
        x=k_decode,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=L,
    )

    q_prefill = q.expand(
        -1,
        -1,
        T_prefill,
        -1,
    )

    q_decode = q.expand(
        -1,
        -1,
        T_decode_append,
        -1,
    )

    scores_direct_prefill = turboquant_polar_prod_inner_product_estimate(
        q=q_prefill,
        encoding=enc_prefill,
        codebooks=codebooks,
        sketch=sketch,
    ).unsqueeze(2)

    scores_direct_decode = turboquant_polar_prod_inner_product_estimate(
        q=q_decode,
        encoding=enc_decode,
        codebooks=codebooks,
        sketch=sketch,
    ).unsqueeze(2)

    scores_direct = torch.cat(
        [
            scores_direct_prefill,
            scores_direct_decode,
        ],
        dim=-1,
    )

    cache_vs_direct = torch.abs(
        scores_cache - scores_direct
    )

    print("========== Cache internal parity ==========")
    print(f"scores_cache.shape = {tuple(scores_cache.shape)}")
    print(f"scores_direct.shape= {tuple(scores_direct.shape)}")
    print(
        f"max_abs_diff       = "
        f"{float(cache_vs_direct.max().item()):.6e}"
    )
    print(
        f"mean_abs_diff      = "
        f"{float(cache_vs_direct.mean().item()):.6e}"
    )
    print()

    assert float(cache_vs_direct.max().item()) < 1e-5

    # ============================================================
    # 7. Compare against dense qK^T quality reference
    # ============================================================

    scores_dense = torch.einsum(
        "bhqd,bhkd->bhqk",
        q,
        k_dense_all,
    )

    quality_metrics = error_metrics(
        pred=scores_cache,
        target=scores_dense,
    )

    print("========== Quantized score quality vs dense qK^T ==========")
    print(f"scores_dense.shape = {tuple(scores_dense.shape)}")
    print(f"MAE                = {quality_metrics['mae']:.6e}")
    print(f"RMSE               = {quality_metrics['rmse']:.6e}")
    print(f"relative RMSE      = {quality_metrics['relative_rmse']:.6e}")
    print(f"max_abs_diff       = {quality_metrics['max_abs_diff']:.6e}")
    print()

    print("[PASS] TurboQuant PolarProd KeyCache reference test passed.")


if __name__ == "__main__":
    main()
