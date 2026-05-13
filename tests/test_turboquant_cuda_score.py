from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.mse_quant import (
    make_random_rotation,
    get_2bit_centroids,
)
from turboquant.qjl import make_gaussian_sketch
from turboquant.key_cache import TurboQuantKeyCache
from turboquant.cuda_score import (
    turboquant_decode_score_cuda_from_cache,
)


@torch.no_grad()
def main():
    torch.manual_seed(0)

    device = "cuda:0"

    B = 1
    H = 32
    T = 64
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

    cache = TurboQuantKeyCache(
        num_layers=1,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    key_states = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=torch.float32,
    )

    query_states = torch.randn(
        B,
        H,
        Q,
        D,
        device=device,
        dtype=torch.float32,
    )

    cache.append(
        layer_idx=0,
        key_states=key_states,
        value_states=None,
    )

    scores_python = cache.score(
        layer_idx=0,
        query_states=query_states,
    )

    scores_cuda = turboquant_decode_score_cuda_from_cache(
        query_states=query_states,
        cache=cache,
        layer_idx=0,
    )

    diff = torch.abs(scores_python - scores_cuda)

    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    print("========== TurboQuant CUDA decode score test ==========")
    print("scores_python.shape:", tuple(scores_python.shape))
    print("scores_cuda.shape:  ", tuple(scores_cuda.shape))
    print(f"max_abs_diff       = {max_abs_diff:.6e}")
    print(f"mean_abs_diff      = {mean_abs_diff:.6e}")

    assert scores_python.shape == scores_cuda.shape
    assert max_abs_diff < 1e-4, f"CUDA score mismatch too large: {max_abs_diff}"

    print("[PASS] TurboQuant CUDA decode score correctness passed.")


if __name__ == "__main__":
    main()
