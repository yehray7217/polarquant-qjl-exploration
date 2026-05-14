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


@torch.no_grad()
def run_case(
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    num_levels: int | None,
) -> None:
    torch.manual_seed(0)

    x = torch.randn(
        *shape,
        device=device,
        dtype=dtype,
    )

    enc = recursive_polar_encode(
        x,
        num_levels=num_levels,
    )

    x_hat = recursive_polar_decode(enc)

    diff = torch.abs(x - x_hat)

    print("------------------------------------------------------------")
    print(f"shape              = {shape}")
    print(f"dtype              = {dtype}")
    print(f"num_levels arg     = {num_levels}")
    print(f"num_angle_levels   = {len(enc.angles)}")
    print(f"radii.shape        = {tuple(enc.radii.shape)}")
    print(f"x_hat.shape        = {tuple(x_hat.shape)}")
    print(f"max_abs_diff       = {float(diff.max().item()):.6e}")
    print(f"mean_abs_diff      = {float(diff.mean().item()):.6e}")

    if dtype == torch.float32:
        assert float(diff.max().item()) < 5e-6
    else:
        assert float(diff.max().item()) < 2e-2


@torch.no_grad()
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("========== PolarQuant recursive polar round-trip ==========")

    # ============================================================
    # Fully recursive: 128 -> 7 levels -> scalar radius
    # ============================================================

    run_case(
        shape=(4, 128),
        dtype=torch.float32,
        device=device,
        num_levels=None,
    )

    run_case(
        shape=(2, 32, 128),
        dtype=torch.float32,
        device=device,
        num_levels=None,
    )

    # ============================================================
    # PolarQuant practical setting: L = 4
    # 128 dims -> 8 remaining radii
    # ============================================================

    run_case(
        shape=(4, 128),
        dtype=torch.float32,
        device=device,
        num_levels=4,
    )

    run_case(
        shape=(2, 32, 128),
        dtype=torch.float32,
        device=device,
        num_levels=4,
    )

    run_case(
        shape=(2, 32, 128),
        dtype=torch.float16,
        device=device,
        num_levels=4,
    )

    print("[PASS] PolarQuant recursive polar round-trip passed.")


if __name__ == "__main__":
    main()