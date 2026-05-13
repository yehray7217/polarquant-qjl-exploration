from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.cuda_packing import (
    pack_2bit_indices_cuda,
    pack_sign_bits_cuda,
)

from turboquant.packing import (
    pack_2bit_indices_unchecked,
    pack_sign_bits_unchecked,
    unpack_2bit_indices,
    unpack_sign_bits,
)


@torch.no_grad()
def main():
    device = "cuda:0"

    torch.manual_seed(0)

    # ============================================================
    # 2-bit index packing parity
    # ============================================================
    indices = torch.randint(
        low=0,
        high=4,
        size=(2, 3, 5, 128),
        dtype=torch.int64,
        device=device,
    )

    packed_ref = pack_2bit_indices_unchecked(indices)
    packed_cuda = pack_2bit_indices_cuda(indices)

    packed_diff = torch.abs(
        packed_ref.to(torch.int16) -
        packed_cuda.to(torch.int16)
    )

    unpacked_cuda = unpack_2bit_indices(
        packed_cuda,
        original_dim=128,
    )

    unpack_diff = torch.abs(
        unpacked_cuda.to(torch.int64) - indices
    )

    print("========== CUDA 2-bit packing test ==========")
    print(f"indices.shape:      {tuple(indices.shape)}")
    print(f"packed_ref.shape:   {tuple(packed_ref.shape)}")
    print(f"packed_cuda.shape:  {tuple(packed_cuda.shape)}")
    print(f"packed max diff:    {int(packed_diff.max().item())}")
    print(f"unpacked max diff:  {int(unpack_diff.max().item())}")

    assert int(packed_diff.max().item()) == 0
    assert int(unpack_diff.max().item()) == 0

    # ============================================================
    # 1-bit sign packing parity
    # ============================================================
    raw = torch.randn(
        2,
        3,
        5,
        256,
        dtype=torch.float32,
        device=device,
    )

    sign_bits = torch.where(
        raw >= 0,
        torch.ones_like(raw),
        -torch.ones_like(raw),
    )

    sign_ref = pack_sign_bits_unchecked(sign_bits)
    sign_cuda = pack_sign_bits_cuda(sign_bits)

    sign_diff = torch.abs(
        sign_ref.to(torch.int16) -
        sign_cuda.to(torch.int16)
    )

    unpacked_sign_cuda = unpack_sign_bits(
        sign_cuda,
        original_dim=256,
        return_pm_one=True,
        dtype=torch.float32,
    )

    unpack_sign_diff = torch.abs(
        unpacked_sign_cuda - sign_bits
    )

    print()
    print("========== CUDA 1-bit sign packing test ==========")
    print(f"sign_bits.shape:        {tuple(sign_bits.shape)}")
    print(f"sign_ref.shape:         {tuple(sign_ref.shape)}")
    print(f"sign_cuda.shape:        {tuple(sign_cuda.shape)}")
    print(f"packed max diff:        {int(sign_diff.max().item())}")
    print(f"unpacked max diff:      {float(unpack_sign_diff.max().item()):.6e}")

    assert int(sign_diff.max().item()) == 0
    assert float(unpack_sign_diff.max().item()) == 0.0

    print()
    print("[PASS] TurboQuant CUDA packing correctness passed.")


if __name__ == "__main__":
    main()