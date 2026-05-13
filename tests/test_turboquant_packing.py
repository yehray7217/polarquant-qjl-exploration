from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.packing import (
    pack_2bit_indices,
    unpack_2bit_indices,
    pack_sign_bits,
    unpack_sign_bits,
)


@torch.no_grad()
def test_pack_unpack_2bit(device: str):
    torch.manual_seed(0)

    B, H, T, D = 2, 3, 5, 128

    indices = torch.randint(
        low=0,
        high=4,
        size=(B, H, T, D),
        dtype=torch.int64,
        device=device,
    )

    packed = pack_2bit_indices(indices)
    unpacked = unpack_2bit_indices(
        packed,
        original_dim=D,
    )

    max_diff = torch.max(torch.abs(indices - unpacked)).item()

    print("========== 2-bit index packing ==========")
    print("indices.shape:", tuple(indices.shape))
    print("packed.shape: ", tuple(packed.shape))
    print("unpacked.shape:", tuple(unpacked.shape))
    print("max diff:", max_diff)

    expected_packed_last_dim = D // 4
    assert packed.dtype == torch.uint8
    assert packed.shape == (B, H, T, expected_packed_last_dim)
    assert torch.equal(indices, unpacked)
    assert max_diff == 0


@torch.no_grad()
def test_pack_unpack_sign_bits(device: str):
    torch.manual_seed(0)

    B, H, T, M = 2, 3, 5, 256

    raw_bits = torch.randint(
        low=0,
        high=2,
        size=(B, H, T, M),
        dtype=torch.int64,
        device=device,
    )

    sign_bits = torch.where(
        raw_bits > 0,
        torch.ones_like(raw_bits, dtype=torch.float32),
        -torch.ones_like(raw_bits, dtype=torch.float32),
    )

    packed = pack_sign_bits(sign_bits)
    unpacked_pm_one = unpack_sign_bits(
        packed,
        original_dim=M,
        return_pm_one=True,
        dtype=torch.float32,
    )

    unpacked_bits = unpack_sign_bits(
        packed,
        original_dim=M,
        return_pm_one=False,
    )

    reconstructed_bits = (unpacked_pm_one > 0).to(torch.int64)

    print()
    print("========== 1-bit sign packing ==========")
    print("sign_bits.shape:", tuple(sign_bits.shape))
    print("packed.shape:   ", tuple(packed.shape))
    print("unpacked.shape: ", tuple(unpacked_pm_one.shape))

    assert packed.dtype == torch.uint8
    assert packed.shape == (B, H, T, M // 8)
    assert torch.equal(raw_bits, unpacked_bits)
    assert torch.equal(raw_bits, reconstructed_bits)


@torch.no_grad()
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    test_pack_unpack_2bit(device)
    test_pack_unpack_sign_bits(device)

    print()
    print("[PASS] TurboQuant packing round-trip test passed.")


if __name__ == "__main__":
    main()
