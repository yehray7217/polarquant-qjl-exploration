from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.polar_packing import (
    pack_polar_angle_codes_l4_d128,
    unpack_polar_angle_codes_l4_d128,
    packed_polar_angle_storage_bytes,
)


@torch.no_grad()
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)

    B = 1
    H = 32
    T = 65

    angle_codes = [
        torch.randint(0, 16, (B, H, T, 64), device=device),
        torch.randint(0, 4, (B, H, T, 32), device=device),
        torch.randint(0, 4, (B, H, T, 16), device=device),
        torch.randint(0, 4, (B, H, T, 8), device=device),
    ]

    packed = pack_polar_angle_codes_l4_d128(
        angle_codes
    )

    unpacked = unpack_polar_angle_codes_l4_d128(
        packed
    )

    print("========== Polar angle packing round-trip ==========")

    for level_idx, (ref, got) in enumerate(
        zip(angle_codes, unpacked),
        start=1,
    ):
        diff = torch.abs(
            ref.to(torch.long) - got.to(torch.long)
        )

        print(
            f"level {level_idx}: "
            f"max_diff={int(diff.max().item())}, "
            f"shape={tuple(got.shape)}"
        )

        assert int(diff.max().item()) == 0

    storage_bytes = packed_polar_angle_storage_bytes(
        packed
    )

    expected_bytes = B * H * T * 46

    print(f"packed storage bytes = {storage_bytes}")
    print(f"expected bytes       = {expected_bytes}")

    assert storage_bytes == expected_bytes

    print("[PASS] Polar angle packing round-trip passed.")


if __name__ == "__main__":
    main()
