from __future__ import annotations

import torch

import turboquant_cuda


@torch.no_grad()
def pack_2bit_indices_cuda(
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA hot-path packer.

    Expected:
      indices: contiguous int64 CUDA tensor [..., D]
      D % 4 == 0

    Returns:
      uint8 CUDA tensor [..., D/4]
    """
    if not indices.is_cuda:
        raise ValueError("indices must be CUDA tensor")

    if indices.dtype != torch.int64:
        raise ValueError(
            f"indices must be torch.int64, got {indices.dtype}"
        )

    if not indices.is_contiguous():
        indices = indices.contiguous()

    return turboquant_cuda.pack_2bit_indices(indices)


@torch.no_grad()
def pack_sign_bits_cuda(
    sign_bits: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA hot-path sign packer.

    Expected:
      sign_bits: contiguous float32 CUDA tensor [..., M]
      M % 8 == 0

    Returns:
      uint8 CUDA tensor [..., M/8]
    """
    if not sign_bits.is_cuda:
        raise ValueError("sign_bits must be CUDA tensor")

    if sign_bits.dtype != torch.float32:
        raise ValueError(
            f"sign_bits must be torch.float32, got {sign_bits.dtype}"
        )

    if not sign_bits.is_contiguous():
        sign_bits = sign_bits.contiguous()

    return turboquant_cuda.pack_sign_bits(sign_bits)
