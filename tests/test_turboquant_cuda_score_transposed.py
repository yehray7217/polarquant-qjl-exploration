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
from turboquant.cuda_score_transposed import (
    turboquant_decode_score_transposed_cuda,
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

    scores_ref = turboquant_decode_score_cuda_from_cache(
        query_states=q.to(torch.float32),
        cache=cache,
        layer_idx=0,
    )

    layer = cache.layers[0]

    packed_mse = layer.packed_mse_indices_buffer[:, :, :T, :]
    packed_qjl = layer.packed_qjl_sign_bits_buffer[:, :, :T, :]

    packed_mse_t = packed_mse.permute(0, 1, 3, 2).contiguous()
    packed_qjl_t = packed_qjl.permute(0, 1, 3, 2).contiguous()

    mse_norms = layer.mse_norms_buffer[:, :, :T].contiguous()
    qjl_norms = layer.qjl_residual_norms_buffer[:, :, :T].contiguous()

    q_flat = q.to(torch.float32).squeeze(2)

    q_rot = q_flat @ rotation.T
    q_sketch = q_flat @ sketch.T

    scores_t = turboquant_decode_score_transposed_cuda(
        q_rot=q_rot,
        q_sketch=q_sketch,
        packed_mse_indices_t=packed_mse_t,
        mse_norms=mse_norms,
        packed_qjl_sign_bits_t=packed_qjl_t,
        qjl_residual_norms=qjl_norms,
        centroids=centroids,
    )

    diff = torch.abs(scores_ref - scores_t)

    print("========== Transposed TurboQuant CUDA score parity ==========")
    print(f"scores_ref.shape = {tuple(scores_ref.shape)}")
    print(f"scores_t.shape   = {tuple(scores_t.shape)}")
    print(f"max_abs_diff     = {float(diff.max().item()):.6e}")
    print(f"mean_abs_diff    = {float(diff.mean().item()):.6e}")

    assert float(diff.max().item()) < 5e-5

    print("[PASS] Transposed TurboQuant CUDA score parity passed.")


if __name__ == "__main__":
    main()
