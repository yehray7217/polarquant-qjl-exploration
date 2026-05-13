from __future__ import annotations

import torch


# ============================================================
# 2-bit centroid index packing
# ============================================================

@torch.no_grad()
def pack_2bit_indices(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack 2-bit indices into uint8.

    Input:
        indices: integer tensor with values in {0,1,2,3}
                 shape [..., D]
                 D must be divisible by 4 for now.

    Output:
        packed: uint8 tensor
                shape [..., D/4]

    Packing layout per byte:
        bits [1:0]   = indices[..., 0]
        bits [3:2]   = indices[..., 1]
        bits [5:4]   = indices[..., 2]
        bits [7:6]   = indices[..., 3]
    """
    if indices.shape[-1] % 4 != 0:
        raise ValueError(
            f"Last dim must be divisible by 4, got {indices.shape[-1]}"
        )

    # Correctness of value range is guaranteed by the quantizer.
    # Do not run GPU->CPU validation checks in the hot decode path.

    x = indices.to(torch.uint8)
    last_dim = x.shape[-1]
    x = x.reshape(*x.shape[:-1], last_dim // 4, 4)

    packed = (
        x[..., 0]
        | (x[..., 1] << 2)
        | (x[..., 2] << 4)
        | (x[..., 3] << 6)
    )

    return packed.contiguous()


@torch.no_grad()
def unpack_2bit_indices(
    packed: torch.Tensor,
    original_dim: int,
) -> torch.Tensor:
    """
    Unpack uint8 back to 2-bit indices.

    Input:
        packed: uint8 tensor shape [..., original_dim/4]
        original_dim: final unpacked D

    Output:
        indices: int64 tensor shape [..., original_dim]
    """
    if original_dim % 4 != 0:
        raise ValueError(f"original_dim must be divisible by 4, got {original_dim}")

    if packed.dtype != torch.uint8:
        raise ValueError(f"packed must be torch.uint8, got {packed.dtype}")

    x0 = packed & 0b00000011
    x1 = (packed >> 2) & 0b00000011
    x2 = (packed >> 4) & 0b00000011
    x3 = (packed >> 6) & 0b00000011

    stacked = torch.stack([x0, x1, x2, x3], dim=-1)
    unpacked = stacked.reshape(*packed.shape[:-1], original_dim)

    return unpacked.to(torch.int64).contiguous()


# ============================================================
# 1-bit sign packing
# ============================================================

@torch.no_grad()
def pack_sign_bits(sign_bits: torch.Tensor) -> torch.Tensor:
    """
    Pack sign bits into uint8.

    Input:
        sign_bits:
            tensor with values in {-1,+1} OR {0,1}
            shape [..., M]
            M must be divisible by 8 for now.

    Output:
        packed:
            uint8 tensor shape [..., M/8]

    Mapping:
        +1 -> 1
        -1 -> 0
    """
    if sign_bits.shape[-1] % 8 != 0:
        raise ValueError(
            f"Last dim must be divisible by 8, got {sign_bits.shape[-1]}"
        )

    bits = (sign_bits > 0).to(torch.uint8)

    last_dim = bits.shape[-1]
    bits = bits.reshape(*bits.shape[:-1], last_dim // 8, 8)

    packed = (
        bits[..., 0]
        | (bits[..., 1] << 1)
        | (bits[..., 2] << 2)
        | (bits[..., 3] << 3)
        | (bits[..., 4] << 4)
        | (bits[..., 5] << 5)
        | (bits[..., 6] << 6)
        | (bits[..., 7] << 7)
    )

    return packed.contiguous()


@torch.no_grad()
def unpack_sign_bits(
    packed: torch.Tensor,
    original_dim: int,
    *,
    return_pm_one: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Unpack uint8 sign storage.

    Input:
        packed: uint8 tensor shape [..., original_dim/8]
        original_dim: final unpacked sign dimension M

    Output:
        if return_pm_one=True:
            float tensor in {-1,+1}, shape [..., M]
        else:
            int64 tensor in {0,1}, shape [..., M]
    """
    if original_dim % 8 != 0:
        raise ValueError(f"original_dim must be divisible by 8, got {original_dim}")

    if packed.dtype != torch.uint8:
        raise ValueError(f"packed must be torch.uint8, got {packed.dtype}")

    bits = [
        (packed >> i) & 0b00000001
        for i in range(8)
    ]

    stacked = torch.stack(bits, dim=-1)
    unpacked = stacked.reshape(*packed.shape[:-1], original_dim)

    if not return_pm_one:
        return unpacked.to(torch.int64).contiguous()

    pm_one = torch.where(
        unpacked > 0,
        torch.ones_like(unpacked, dtype=dtype),
        -torch.ones_like(unpacked, dtype=dtype),
    )

    return pm_one.contiguous()

@torch.no_grad()
def pack_2bit_indices_unchecked(indices: torch.Tensor) -> torch.Tensor:
    """
    Runtime hot-path version.
    Assumes:
      - last dim divisible by 4
      - values already in {0,1,2,3}
    Avoids CUDA tensor validation that causes synchronization.
    """
    x = indices.to(torch.uint8)
    last_dim = x.shape[-1]
    x = x.reshape(*x.shape[:-1], last_dim // 4, 4)

    packed = (
        x[..., 0]
        | (x[..., 1] << 2)
        | (x[..., 2] << 4)
        | (x[..., 3] << 6)
    )

    return packed.contiguous()


@torch.no_grad()
def pack_sign_bits_unchecked(sign_bits: torch.Tensor) -> torch.Tensor:
    """
    Runtime hot-path version.
    Assumes sign_bits is float-like {-1,+1}.
    Avoids min/max/item checks.
    """
    bits = (sign_bits > 0).to(torch.uint8)

    last_dim = bits.shape[-1]
    bits = bits.reshape(*bits.shape[:-1], last_dim // 8, 8)

    packed = (
        bits[..., 0]
        | (bits[..., 1] << 1)
        | (bits[..., 2] << 2)
        | (bits[..., 3] << 3)
        | (bits[..., 4] << 4)
        | (bits[..., 5] << 5)
        | (bits[..., 6] << 6)
        | (bits[..., 7] << 7)
    )

    return packed.contiguous()