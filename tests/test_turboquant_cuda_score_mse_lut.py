from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.key_cache import TurboQuantKeyCache
from turboquant.mse_quant import (
    make_random_rotation,
    get_2bit_centroids,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)
from turboquant.cuda_score import (
    turboquant_decode_score_cuda_from_cache,
)
from turboquant.cuda_score_mse_lut import (
    turboquant_mse_lut_score_transposed_cuda,
)


@torch.no_grad()
def main():
    device = "cuda:0"
    dtype = torch.float16

    B = 1
    H = 32
    T = 4096
    D = 128
    M = 256

    torch.manual_seed(0)

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

    cache = TurboQuantKeyCache(
        num_layers=1,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
        max_cache_len=T,
    )

    k = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=dtype,
    )

    cache.append(
        layer_idx=0,
        key_states=k,
        value_states=None,
    )

    q = torch.randn(
        B,
        H,
        1,
        D,
        device=device,
        dtype=dtype,
    )

    q_fp32 = q.to(torch.float32)

    # ============================================================
    # Reference:
    # existing full prod score kernel with QJL residual zeroed.
    # This yields MSE-only score.
    # ============================================================

    layer = cache.layers[0]

    qjl_backup = layer.qjl_residual_norms_buffer[:, :, :T].clone()

    layer.qjl_residual_norms_buffer[:, :, :T].zero_()

    scores_ref = turboquant_decode_score_cuda_from_cache(
            query_states=q_fp32,
            cache=cache,
            layer_idx=0,
        )

    layer.qjl_residual_norms_buffer[:, :, :T].copy_(
        qjl_backup
    )

    # ============================================================
    # LUT kernel input
    # ============================================================

    q_flat = q_fp32.squeeze(2)

    q_rot = q_flat @ rotation.T

    packed_mse = layer.packed_mse_indices_buffer[
            :, :, :T, :
        ]

    packed_mse_t = packed_mse.permute(
            0, 1, 3, 2
        ).contiguous()

    mse_norms = layer.mse_norms_buffer[
            :, :, :T
        ].contiguous()

    scores_lut = turboquant_mse_lut_score_transposed_cuda(
            q_rot=q_rot,
            packed_mse_indices_t=packed_mse_t,
            mse_norms=mse_norms,
            centroids=centroids,
        )

    diff = torch.abs(
            scores_ref - scores_lut
        )

    print("========== TurboQuant 2-bit MSE LUT score parity ==========")
    print(f"scores_ref.shape = {tuple(scores_ref.shape)}")
    print(f"scores_lut.shape = {tuple(scores_lut.shape)}")
    print(f"max_abs_diff     = {float(diff.max().item()):.6e}")
    print(f"mean_abs_diff    = {float(diff.mean().item()):.6e}")

    assert float(diff.max().item()) < 5e-5

    print("[PASS] TurboQuant 2-bit MSE LUT score parity passed.")


if __name__ == "__main__":
    main()
