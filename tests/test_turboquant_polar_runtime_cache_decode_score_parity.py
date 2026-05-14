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
from turboquant.polar_runtime_cache import (
    TurboQuantPolarRuntimeCache,
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

    target_rms = torch.sqrt(
        torch.mean(target_f * target_f)
    )

    relative_rmse = rmse / torch.clamp(
        target_rms,
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

    num_layers = 1

    T_prefill = 64
    T_decode_append = 1
    T_total = T_prefill + T_decode_append

    N_calib = 4096

    # ============================================================
    # 1. Fit Polar angle codebooks
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

    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    # ============================================================
    # 2. Create runtime cache
    # ============================================================

    runtime_cache = TurboQuantPolarRuntimeCache(
        num_layers=num_layers,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=L,
    )

    # ============================================================
    # 3. Build K/V chunks
    # ============================================================

    k_prefill = torch.randn(
        B,
        H,
        T_prefill,
        D,
        device=device,
        dtype=dtype,
    )

    v_prefill = torch.randn(
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

    v_decode = torch.randn(
        B,
        H,
        T_decode_append,
        D,
        device=device,
        dtype=dtype,
    )

    k_all = torch.cat(
        [k_prefill, k_decode],
        dim=2,
    )

    # ============================================================
    # 4. Update runtime cache
    # ============================================================

    runtime_cache.update(
        k_prefill,
        v_prefill,
        layer_idx=0,
    )

    runtime_cache.update(
        k_decode,
        v_decode,
        layer_idx=0,
    )

    assert runtime_cache.get_seq_length(0) == T_total
    assert runtime_cache.seen_tokens == T_total

    # ============================================================
    # 5. Decode-style query
    # ============================================================

    q = torch.randn(
        B,
        H,
        1,
        D,
        device=device,
        dtype=dtype,
    )

    # ============================================================
    # 6. Runtime cache score
    # ============================================================

    scores_runtime = runtime_cache.score(
        layer_idx=0,
        query_states=q,
    )

    assert tuple(scores_runtime.shape) == (
        B,
        H,
        1,
        T_total,
    )

    # ============================================================
    # 7. Direct PolarProd score reference
    #    Build compressed encodings chunk-by-chunk exactly as cache did.
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

    scores_prefill_direct = turboquant_polar_prod_inner_product_estimate(
        q=q_prefill,
        encoding=enc_prefill,
        codebooks=codebooks,
        sketch=sketch,
    ).unsqueeze(2)

    scores_decode_direct = turboquant_polar_prod_inner_product_estimate(
        q=q_decode,
        encoding=enc_decode,
        codebooks=codebooks,
        sketch=sketch,
    ).unsqueeze(2)

    scores_direct = torch.cat(
        [
            scores_prefill_direct,
            scores_decode_direct,
        ],
        dim=-1,
    )

    # ============================================================
    # 8. Runtime cache parity
    # ============================================================

    parity_diff = torch.abs(
        scores_runtime - scores_direct
    )

    print("========== TurboQuant Polar runtime decode score parity ==========")
    print(f"scores_runtime.shape = {tuple(scores_runtime.shape)}")
    print(f"scores_direct.shape  = {tuple(scores_direct.shape)}")
    print(
        f"max_abs_diff         = "
        f"{float(parity_diff.max().item()):.6e}"
    )
    print(
        f"mean_abs_diff        = "
        f"{float(parity_diff.mean().item()):.6e}"
    )
    print()

    assert float(parity_diff.max().item()) < 1e-5

    # ============================================================
    # 9. Quality vs dense qK^T
    # ============================================================

    scores_dense = torch.einsum(
        "bhqd,bhkd->bhqk",
        q,
        k_all,
    )

    quality = error_metrics(
        pred=scores_runtime,
        target=scores_dense,
    )

    print("========== Quantized runtime score quality vs dense qK^T ==========")
    print(f"scores_dense.shape   = {tuple(scores_dense.shape)}")
    print(f"MAE                  = {quality['mae']:.6e}")
    print(f"RMSE                 = {quality['rmse']:.6e}")
    print(f"relative RMSE        = {quality['relative_rmse']:.6e}")
    print(f"max_abs_diff         = {quality['max_abs_diff']:.6e}")
    print()

    # ============================================================
    # 10. Dense V cache sanity
    # ============================================================

    v_all_expected = torch.cat(
        [v_prefill, v_decode],
        dim=2,
    )

    v_all_runtime = runtime_cache.get_value_states(
        layer_idx=0,
    )

    v_diff = torch.abs(
        v_all_runtime - v_all_expected
    )

    print("========== Dense V runtime cache sanity ==========")
    print(f"v_all_runtime.shape  = {tuple(v_all_runtime.shape)}")
    print(
        f"max_abs_diff         = "
        f"{float(v_diff.max().item()):.6e}"
    )
    print(
        f"mean_abs_diff        = "
        f"{float(v_diff.mean().item()):.6e}"
    )
    print()

    assert float(v_diff.max().item()) == 0.0

    print("[PASS] TurboQuant Polar runtime decode score parity test passed.")


if __name__ == "__main__":
    main()
