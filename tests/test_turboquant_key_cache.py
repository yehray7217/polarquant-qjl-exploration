from pathlib import Path
import sys
import math

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
    turboquant_prod_inner_product_estimate,
)
from turboquant.key_cache import TurboQuantKeyCache


@torch.no_grad()
def direct_turboquant_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
    sketch: torch.Tensor,
) -> torch.Tensor:
    """
    Reference path:
      fp32 K -> on-the-fly TurboQuant encode -> estimate score
    """
    B, H, Q, D = q.shape
    _, _, T, _ = k.shape

    k_flat = k.float().reshape(-1, D)

    enc = turboquant_prod_quantize_3bit(
        x=k_flat,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    chunks = []

    for q_idx in range(Q):
        q_one = q[:, :, q_idx:q_idx + 1, :]

        q_repeat = (
            q_one.float()
            .reshape(B * H, 1, D)
            .expand(-1, T, -1)
            .reshape(B * H * T, D)
        )

        est_dot = turboquant_prod_inner_product_estimate(
            q=q_repeat,
            encoding=enc,
            rotation=rotation,
            centroids=centroids,
            sketch=sketch,
        )

        chunks.append(est_dot.reshape(B, H, 1, T))

    return torch.cat(chunks, dim=2)


@torch.no_grad()
def main():
    torch.manual_seed(0)

    device = "cuda:0"

    B = 1
    H = 32
    T0 = 64
    T1 = 1
    Q = 1
    D = 128
    M = 256

    rotation = make_random_rotation(
        d=D,
        device=device,
        dtype=torch.float32,
        seed=123,
    )
    centroids = get_2bit_centroids(
        d=D,
        device=device,
        dtype=torch.float32,
    )
    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=456,
    )

    # ------------------------------------------------------------
    # Initial prefill-like cache append
    # ------------------------------------------------------------
    k0 = torch.randn(B, H, T0, D, device=device, dtype=torch.float32)
    v0 = torch.randn(B, H, T0, D, device=device, dtype=torch.float16)
    q0 = torch.randn(B, H, Q, D, device=device, dtype=torch.float32)

    cache = TurboQuantKeyCache(
        num_layers=1,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    cache.append(
        layer_idx=0,
        key_states=k0,
        value_states=v0,
    )

    print("========== After first append ==========")
    print("seq_len:", cache.get_seq_length(0))
    print(cache.report(0))

    assert cache.get_seq_length(0) == T0

    # ------------------------------------------------------------
    # Compare score from compressed cache vs direct reference path
    # ------------------------------------------------------------
    scores_ref = direct_turboquant_scores(
        q=q0,
        k=k0,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    scores_cache = cache.score(
        layer_idx=0,
        query_states=q0,
    )

    max_abs_diff = torch.max(torch.abs(scores_ref - scores_cache)).item()
    mean_abs_diff = torch.mean(torch.abs(scores_ref - scores_cache)).item()

    print()
    print("========== Score equality check ==========")
    print("scores_ref.shape:  ", tuple(scores_ref.shape))
    print("scores_cache.shape:", tuple(scores_cache.shape))
    print(f"max_abs_diff       = {max_abs_diff:.6e}")
    print(f"mean_abs_diff      = {mean_abs_diff:.6e}")

    assert scores_ref.shape == scores_cache.shape
    assert max_abs_diff < 1e-5, f"score mismatch too large: {max_abs_diff}"

    # ------------------------------------------------------------
    # Append one decode-like token
    # ------------------------------------------------------------
    k1 = torch.randn(B, H, T1, D, device=device, dtype=torch.float32)
    v1 = torch.randn(B, H, T1, D, device=device, dtype=torch.float16)
    q1 = torch.randn(B, H, Q, D, device=device, dtype=torch.float32)

    cache.append(
        layer_idx=0,
        key_states=k1,
        value_states=v1,
    )

    print()
    print("========== After decode-token append ==========")
    print("seq_len:", cache.get_seq_length(0))
    print(cache.report(0))

    assert cache.get_seq_length(0) == T0 + T1

    k_full = torch.cat([k0, k1], dim=2)

    scores_ref_after_append = direct_turboquant_scores(
        q=q1,
        k=k_full,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    scores_cache_after_append = cache.score(
        layer_idx=0,
        query_states=q1,
    )

    max_abs_diff_append = torch.max(
        torch.abs(scores_ref_after_append - scores_cache_after_append)
    ).item()

    mean_abs_diff_append = torch.mean(
        torch.abs(scores_ref_after_append - scores_cache_after_append)
    ).item()

    print()
    print("========== Post-append score equality check ==========")
    print("scores_ref_after_append.shape:  ", tuple(scores_ref_after_append.shape))
    print("scores_cache_after_append.shape:", tuple(scores_cache_after_append.shape))
    print(f"max_abs_diff                   = {max_abs_diff_append:.6e}")
    print(f"mean_abs_diff                  = {mean_abs_diff_append:.6e}")

    assert scores_ref_after_append.shape == scores_cache_after_append.shape
    assert max_abs_diff_append < 1e-5, (
        f"post-append score mismatch too large: {max_abs_diff_append}"
    )

    # ------------------------------------------------------------
    # V-cache shape sanity
    # ------------------------------------------------------------
    values = cache.get_value_states(0)
    print()
    print("========== Value-state shape sanity ==========")
    print("V shape:", tuple(values.shape))

    assert tuple(values.shape) == (B, H, T0 + T1, D)

    print()
    print("[PASS] TurboQuantKeyCache correctness test passed.")


if __name__ == "__main__":
    main()
