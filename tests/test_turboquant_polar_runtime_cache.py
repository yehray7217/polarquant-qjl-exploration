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
from turboquant.polar_runtime_cache import (
    TurboQuantPolarRuntimeCache,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)


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
    M = 256

    num_layers = 2

    T_prefill = 64
    T_decode = 1
    T_total = T_prefill + T_decode

    N_calib = 4096

    # ============================================================
    # 1. Fit polar angle codebooks
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

    cache = TurboQuantPolarRuntimeCache(
        num_layers=num_layers,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=L,
    )

    # ============================================================
    # 3. Prefill update for both layers
    # ============================================================

    layer_0_k_prefill = torch.randn(
        B, H, T_prefill, D,
        device=device,
        dtype=dtype,
    )
    layer_0_v_prefill = torch.randn(
        B, H, T_prefill, D,
        device=device,
        dtype=dtype,
    )

    layer_1_k_prefill = torch.randn(
        B, H, T_prefill, D,
        device=device,
        dtype=dtype,
    )
    layer_1_v_prefill = torch.randn(
        B, H, T_prefill, D,
        device=device,
        dtype=dtype,
    )

    _, layer_0_v_full = cache.update(
        layer_0_k_prefill,
        layer_0_v_prefill,
        layer_idx=0,
    )

    _, layer_1_v_full = cache.update(
        layer_1_k_prefill,
        layer_1_v_prefill,
        layer_idx=1,
    )

    assert tuple(layer_0_v_full.shape) == (
        B, H, T_prefill, D
    )
    assert tuple(layer_1_v_full.shape) == (
        B, H, T_prefill, D
    )

    assert cache.get_seq_length(0) == T_prefill
    assert cache.get_seq_length(1) == T_prefill
    assert cache.seen_tokens == T_prefill

    print("========== After prefill update ==========")
    print(cache.report())
    print()

    # ============================================================
    # 4. Decode-token update for both layers
    # ============================================================

    layer_0_k_decode = torch.randn(
        B, H, T_decode, D,
        device=device,
        dtype=dtype,
    )
    layer_0_v_decode = torch.randn(
        B, H, T_decode, D,
        device=device,
        dtype=dtype,
    )

    layer_1_k_decode = torch.randn(
        B, H, T_decode, D,
        device=device,
        dtype=dtype,
    )
    layer_1_v_decode = torch.randn(
        B, H, T_decode, D,
        device=device,
        dtype=dtype,
    )

    _, layer_0_v_full = cache.update(
        layer_0_k_decode,
        layer_0_v_decode,
        layer_idx=0,
    )

    _, layer_1_v_full = cache.update(
        layer_1_k_decode,
        layer_1_v_decode,
        layer_idx=1,
    )

    assert tuple(layer_0_v_full.shape) == (
        B, H, T_total, D
    )
    assert tuple(layer_1_v_full.shape) == (
        B, H, T_total, D
    )

    assert cache.get_seq_length(0) == T_total
    assert cache.get_seq_length(1) == T_total
    assert cache.seen_tokens == T_total

    print("========== After decode update ==========")
    print(cache.report())
    print()

    # ============================================================
    # 5. Score shape sanity
    # ============================================================

    q_layer_0 = torch.randn(
        B, H, 1, D,
        device=device,
        dtype=dtype,
    )

    scores_layer_0 = cache.score(
        layer_idx=0,
        query_states=q_layer_0,
    )

    assert tuple(scores_layer_0.shape) == (
        B, H, 1, T_total
    )

    q_layer_1 = torch.randn(
        B, H, 1, D,
        device=device,
        dtype=dtype,
    )

    scores_layer_1 = cache.score(
        layer_idx=1,
        query_states=q_layer_1,
    )

    assert tuple(scores_layer_1.shape) == (
        B, H, 1, T_total
    )

    print("========== Score shape sanity ==========")
    print(f"scores_layer_0.shape = {tuple(scores_layer_0.shape)}")
    print(f"scores_layer_1.shape = {tuple(scores_layer_1.shape)}")
    print()

    # ============================================================
    # 6. Value cache sanity
    # ============================================================

    value_layer_0 = cache.get_value_states(
        layer_idx=0,
    )
    value_layer_1 = cache.get_value_states(
        layer_idx=1,
    )

    assert tuple(value_layer_0.shape) == (
        B, H, T_total, D
    )
    assert tuple(value_layer_1.shape) == (
        B, H, T_total, D
    )

    print("========== Value cache sanity ==========")
    print(f"value_layer_0.shape = {tuple(value_layer_0.shape)}")
    print(f"value_layer_1.shape = {tuple(value_layer_1.shape)}")
    print()

    print("[PASS] TurboQuant Polar runtime cache test passed.")


if __name__ == "__main__":
    main()
