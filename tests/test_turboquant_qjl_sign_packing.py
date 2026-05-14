from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.qjl_packing import (
    pack_qjl_signs_1bit,
    unpack_qjl_signs_1bit,
    packed_qjl_sign_storage_bytes,
)


@torch.no_grad()
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)

    N = 4096
    M = 256

    raw = torch.randn(
        N,
        M,
        device=device,
        dtype=torch.float32,
    )

    signs_ref = raw >= 0

    packed = pack_qjl_signs_1bit(
        signs_ref
    )

    signs_unpacked = unpack_qjl_signs_1bit(
        packed
    )

    diff = (
        signs_ref.to(torch.int32) -
        signs_unpacked.to(torch.int32)
    ).abs()

    storage_bytes = packed_qjl_sign_storage_bytes(
        packed
    )

    expected_bytes = N * (M // 8)

    print("========== QJL 1-bit sign packing round-trip ==========")
    print(f"signs_ref.shape       = {tuple(signs_ref.shape)}")
    print(f"packed.shape          = {tuple(packed.shape)}")
    print(f"signs_unpacked.shape  = {tuple(signs_unpacked.shape)}")
    print(f"max_abs_diff          = {float(diff.max().item()):.6e}")
    print(f"mean_abs_diff         = {float(diff.to(torch.float32).mean().item()):.6e}")
    print(f"packed storage bytes  = {storage_bytes}")
    print(f"expected bytes        = {expected_bytes}")

    assert float(diff.max().item()) == 0.0
    assert storage_bytes == expected_bytes

    print("[PASS] QJL 1-bit sign packing round-trip passed.")


if __name__ == "__main__":
    main()
