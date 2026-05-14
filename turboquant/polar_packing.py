from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PackedPolarAngles:
    """
    Packed PolarQuant angle codes for L=4, D=128 configuration.

    level1_4bit:
        [..., 32] uint8
        Two 4-bit codes per byte.

    level2_2bit:
        [..., 8] uint8
        Four 2-bit codes per byte.

    level3_2bit:
        [..., 4] uint8

    level4_2bit:
        [..., 2] uint8
    """
    level1_4bit: torch.Tensor
    level2_2bit: torch.Tensor
    level3_2bit: torch.Tensor
    level4_2bit: torch.Tensor


def _pack_4bit_codes(
    codes: torch.Tensor,
) -> torch.Tensor:
    """
    codes: [..., N], N even, values 0..15
    out:   [..., N/2] uint8
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError("4-bit code count must be even.")

    codes_u8 = codes.to(torch.uint8)

    low = codes_u8[..., 0::2]
    high = codes_u8[..., 1::2]

    return low | (high << 4)


def _unpack_4bit_codes(
    packed: torch.Tensor,
) -> torch.Tensor:
    """
    packed: [..., N/2] uint8
    out:    [..., N] int64
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    return torch.stack(
        [low, high],
        dim=-1,
    ).flatten(start_dim=-2).to(torch.long)


def _pack_2bit_codes(
    codes: torch.Tensor,
) -> torch.Tensor:
    """
    codes: [..., N], N divisible by 4, values 0..3
    out:   [..., N/4] uint8
    """
    if codes.shape[-1] % 4 != 0:
        raise ValueError("2-bit code count must be divisible by 4.")

    codes_u8 = codes.to(torch.uint8)

    c0 = codes_u8[..., 0::4]
    c1 = codes_u8[..., 1::4]
    c2 = codes_u8[..., 2::4]
    c3 = codes_u8[..., 3::4]

    return (
        c0 |
        (c1 << 2) |
        (c2 << 4) |
        (c3 << 6)
    )


def _unpack_2bit_codes(
    packed: torch.Tensor,
) -> torch.Tensor:
    """
    packed: [..., N/4] uint8
    out:    [..., N] int64
    """
    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03

    return torch.stack(
        [c0, c1, c2, c3],
        dim=-1,
    ).flatten(start_dim=-2).to(torch.long)


@torch.no_grad()
def pack_polar_angle_codes_l4_d128(
    angle_codes: list[torch.Tensor],
) -> PackedPolarAngles:
    """
    Expected shapes for D=128, L=4:
      level 1: [..., 64]
      level 2: [..., 32]
      level 3: [..., 16]
      level 4: [..., 8]
    """
    if len(angle_codes) != 4:
        raise ValueError(
            f"Expected 4 angle levels, got {len(angle_codes)}."
        )

    l1, l2, l3, l4 = angle_codes

    if l1.shape[-1] != 64:
        raise ValueError(f"Level 1 expected 64 codes, got {l1.shape[-1]}.")
    if l2.shape[-1] != 32:
        raise ValueError(f"Level 2 expected 32 codes, got {l2.shape[-1]}.")
    if l3.shape[-1] != 16:
        raise ValueError(f"Level 3 expected 16 codes, got {l3.shape[-1]}.")
    if l4.shape[-1] != 8:
        raise ValueError(f"Level 4 expected 8 codes, got {l4.shape[-1]}.")

    return PackedPolarAngles(
        level1_4bit=_pack_4bit_codes(l1),
        level2_2bit=_pack_2bit_codes(l2),
        level3_2bit=_pack_2bit_codes(l3),
        level4_2bit=_pack_2bit_codes(l4),
    )


@torch.no_grad()
def unpack_polar_angle_codes_l4_d128(
    packed: PackedPolarAngles,
) -> list[torch.Tensor]:
    return [
        _unpack_4bit_codes(packed.level1_4bit),
        _unpack_2bit_codes(packed.level2_2bit),
        _unpack_2bit_codes(packed.level3_2bit),
        _unpack_2bit_codes(packed.level4_2bit),
    ]


def packed_polar_angle_storage_bytes(
    packed: PackedPolarAngles,
) -> int:
    return int(
        packed.level1_4bit.numel() * packed.level1_4bit.element_size()
        + packed.level2_2bit.numel() * packed.level2_2bit.element_size()
        + packed.level3_2bit.numel() * packed.level3_2bit.element_size()
        + packed.level4_2bit.numel() * packed.level4_2bit.element_size()
    )
