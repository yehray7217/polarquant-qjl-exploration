from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.packed_meta import (
    PACKED_META_BYTES,
    build_turboquant_packed_meta_blob,
    unpack_turboquant_packed_meta_blob,
    packed_meta_storage_bytes,
)


@torch.no_grad()
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)

    B = 1
    H = 32
    T = 65

    packed_l1 = torch.randint(
        0,
        256,
        (B, H, T, 32),
        device=device,
        dtype=torch.uint8,
    )

    packed_l2 = torch.randint(
        0,
        256,
        (B, H, T, 8),
        device=device,
        dtype=torch.uint8,
    )

    packed_l3 = torch.randint(
        0,
        256,
        (B, H, T, 4),
        device=device,
        dtype=torch.uint8,
    )

    packed_l4 = torch.randint(
        0,
        256,
        (B, H, T, 2),
        device=device,
        dtype=torch.uint8,
    )

    packed_qjl_signs = torch.randint(
        0,
        256,
        (B, H, T, 16),
        device=device,
        dtype=torch.uint8,
    )

    packed_meta = build_turboquant_packed_meta_blob(
        packed_l1=packed_l1,
        packed_l2=packed_l2,
        packed_l3=packed_l3,
        packed_l4=packed_l4,
        packed_qjl_signs=packed_qjl_signs,
    )

    unpacked = unpack_turboquant_packed_meta_blob(
        packed_meta
    )

    print("========== TurboQuant packed-meta blob layout test ==========")
    print(f"packed_meta.shape         = {tuple(packed_meta.shape)}")
    print(f"packed_meta.dtype         = {packed_meta.dtype}")
    print(f"packed_meta last dim      = {packed_meta.shape[-1]}")
    print(f"storage bytes             = {packed_meta_storage_bytes(packed_meta)}")
    print()

    checks = [
        ("packed_l1", packed_l1),
        ("packed_l2", packed_l2),
        ("packed_l3", packed_l3),
        ("packed_l4", packed_l4),
        ("packed_qjl_signs", packed_qjl_signs),
    ]

    for name, ref in checks:
        got = unpacked[name]

        diff = (
            ref.to(torch.int16)
            - got.to(torch.int16)
        ).abs()

        max_diff = int(diff.max().item())

        print(
            f"{name:20s} "
            f"shape={tuple(got.shape)} "
            f"max_diff={max_diff}"
        )

        assert max_diff == 0

    padding = unpacked["padding"]
    padding_max = int(
        padding.to(torch.int16).abs().max().item()
    )

    print(
        f"{'padding':20s} "
        f"shape={tuple(padding.shape)} "
        f"max_abs={padding_max}"
    )

    assert padding_max == 0
    assert tuple(packed_meta.shape) == (
        B,
        H,
        T,
        PACKED_META_BYTES,
    )

    expected_bytes = (
        B
        * H
        * T
        * PACKED_META_BYTES
    )

    assert packed_meta_storage_bytes(packed_meta) == expected_bytes

    print("[PASS] TurboQuant packed-meta blob layout passed.")


if __name__ == "__main__":
    main()
