from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.polar_packing_stage1_3bit import (
    PAPER_4BIT_STAGE1_BITS_BY_LEVEL,
    pack_polar_angle_codes_l4_d128_stage1_3bit,
    unpack_polar_angle_codes_l4_d128_stage1_3bit,
    packed_polar_angle_stage1_3bit_storage_bytes,
    stage1_3bit_angle_bytes_per_vector_l4_d128,
    stage1_3bit_angle_bits_per_channel_l4_d128,
)


@torch.no_grad()
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)

    B = 1
    H = 32
    T = 65

    angle_codes = [
        torch.randint(
            0,
            16,
            (B, H, T, 64),
            device=device,
            dtype=torch.long,
        ),
        torch.randint(
            0,
            4,
            (B, H, T, 32),
            device=device,
            dtype=torch.long,
        ),
        torch.randint(
            0,
            8,
            (B, H, T, 16),
            device=device,
            dtype=torch.long,
        ),
        torch.randint(
            0,
            4,
            (B, H, T, 8),
            device=device,
            dtype=torch.long,
        ),
    ]

    packed = pack_polar_angle_codes_l4_d128_stage1_3bit(
        angle_codes
    )

    unpacked = unpack_polar_angle_codes_l4_d128_stage1_3bit(
        packed
    )

    print("========== Polar Stage-1 3-bit angle packing round-trip ==========")
    print(f"bits_by_level         = {PAPER_4BIT_STAGE1_BITS_BY_LEVEL}")

    for level_idx, (ref, got) in enumerate(
        zip(angle_codes, unpacked),
        start=1,
    ):
        diff = (
            ref.to(torch.int64)
            - got.to(torch.int64)
        ).abs()

        print(
            f"level {level_idx}: "
            f"max_diff={int(diff.max().item())}, "
            f"shape={tuple(got.shape)}"
        )

        assert int(diff.max().item()) == 0

    storage_bytes = packed_polar_angle_stage1_3bit_storage_bytes(
        packed
    )

    expected_bytes_per_vector = (
        stage1_3bit_angle_bytes_per_vector_l4_d128()
    )

    expected_storage_bytes = (
        B * H * T * expected_bytes_per_vector
    )

    print()
    print(f"packed storage bytes  = {storage_bytes}")
    print(f"expected bytes        = {expected_storage_bytes}")
    print(
        "angle bits/channel    = "
        f"{stage1_3bit_angle_bits_per_channel_l4_d128():.6f}"
    )

    assert storage_bytes == expected_storage_bytes
    assert expected_bytes_per_vector == 48
    assert stage1_3bit_angle_bits_per_channel_l4_d128() == 3.0

    print("[PASS] Polar Stage-1 3-bit angle packing round-trip passed.")


if __name__ == "__main__":
    main()
