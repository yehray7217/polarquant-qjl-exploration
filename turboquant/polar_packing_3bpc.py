from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PackedPolarAngles3Bpc:
    """
    PolarQuant D=128, L=4 packed angles for the ~3 bpc aligned experiment.

    bits_by_level = (2,1,1,1)

    level1_2bit:
        [..., 16] uint8
        Four 2-bit codes per byte.

    level2_1bit:
        [..., 4] uint8
        Eight 1-bit codes per byte.

    level3_1bit:
        [..., 2] uint8

    level4_1bit:
        [..., 1] uint8
    """
    level1_2bit: torch.Tensor
    level2_1bit: torch.Tensor
    level3_1bit: torch.Tensor
    level4_1bit: torch.Tensor


def _pack_2bit_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 4 != 0:
        raise ValueError("2-bit code count must be divisible by 4.")

    codes_u8 = codes.to(torch.uint8)
    c0 = codes_u8[..., 0::4]
    c1 = codes_u8[..., 1::4]
    c2 = codes_u8[..., 2::4]
    c3 = codes_u8[..., 3::4]

    return (
        c0
        | (c1 << 2)
        | (c2 << 4)
        | (c3 << 6)
    ).contiguous()


def _unpack_2bit_codes(packed: torch.Tensor) -> torch.Tensor:
    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03

    return torch.stack(
        [c0, c1, c2, c3],
        dim=-1,
    ).flatten(start_dim=-2).to(torch.long).contiguous()


def _pack_1bit_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 8 != 0:
        raise ValueError("1-bit code count must be divisible by 8.")

    codes_u8 = codes.to(torch.uint8) & 0x01
    chunks = [codes_u8[..., i::8] << i for i in range(8)]
    out = chunks[0]
    for c in chunks[1:]:
        out = out | c
    return out.contiguous()


def _unpack_1bit_codes(packed: torch.Tensor) -> torch.Tensor:
    bits = [
        ((packed >> i) & 0x01)
        for i in range(8)
    ]
    return torch.stack(bits, dim=-1).flatten(start_dim=-2).to(torch.long).contiguous()


@torch.no_grad()
def pack_polar_angle_codes_3bpc_l4_d128(
    angle_codes: list[torch.Tensor],
) -> PackedPolarAngles3Bpc:
    """
    Expected code shapes for D=128, L=4:
      level 1: [..., 64], 2 bits
      level 2: [..., 32], 1 bit
      level 3: [..., 16], 1 bit
      level 4: [...,  8], 1 bit
    """
    if len(angle_codes) != 4:
        raise ValueError(f"Expected 4 angle levels, got {len(angle_codes)}.")

    l1, l2, l3, l4 = angle_codes

    if l1.shape[-1] != 64:
        raise ValueError(f"Level 1 expected 64 codes, got {l1.shape[-1]}.")
    if l2.shape[-1] != 32:
        raise ValueError(f"Level 2 expected 32 codes, got {l2.shape[-1]}.")
    if l3.shape[-1] != 16:
        raise ValueError(f"Level 3 expected 16 codes, got {l3.shape[-1]}.")
    if l4.shape[-1] != 8:
        raise ValueError(f"Level 4 expected 8 codes, got {l4.shape[-1]}.")

    return PackedPolarAngles3Bpc(
        level1_2bit=_pack_2bit_codes(l1),
        level2_1bit=_pack_1bit_codes(l2),
        level3_1bit=_pack_1bit_codes(l3),
        level4_1bit=_pack_1bit_codes(l4),
    )


@torch.no_grad()
def unpack_polar_angle_codes_3bpc_l4_d128(
    packed: PackedPolarAngles3Bpc,
) -> list[torch.Tensor]:
    return [
        _unpack_2bit_codes(packed.level1_2bit),
        _unpack_1bit_codes(packed.level2_1bit),
        _unpack_1bit_codes(packed.level3_1bit),
        _unpack_1bit_codes(packed.level4_1bit),
    ]


def packed_polar_angle_storage_bytes_3bpc(
    packed: PackedPolarAngles3Bpc,
) -> int:
    return int(
        packed.level1_2bit.numel() * packed.level1_2bit.element_size()
        + packed.level2_1bit.numel() * packed.level2_1bit.element_size()
        + packed.level3_1bit.numel() * packed.level3_1bit.element_size()
        + packed.level4_1bit.numel() * packed.level4_1bit.element_size()
    )
