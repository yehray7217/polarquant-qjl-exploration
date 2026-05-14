from __future__ import annotations

from dataclasses import dataclass

import torch


PAPER_4BIT_STAGE1_BITS_BY_LEVEL: tuple[int, int, int, int] = (
    4,
    2,
    3,
    2,
)


@dataclass
class PackedPolarAnglesStage1ThreeBit:
    """
    Packed Polar Stage-1 angle codes for the paper-budget-aligned
    3-bit Stage 1 configuration:

        bits_by_level = (4, 2, 3, 2)

    For D=128 and L=4:

        level1_4bit:
            [..., 32] uint8
            64 angle codes × 4 bits

        level2_2bit:
            [..., 8] uint8
            32 angle codes × 2 bits

        level3_3bit:
            [..., 6] uint8
            16 angle codes × 3 bits

        level4_2bit:
            [..., 2] uint8
            8 angle codes × 2 bits
    """
    level1_4bit: torch.Tensor
    level2_2bit: torch.Tensor
    level3_3bit: torch.Tensor
    level4_2bit: torch.Tensor


# ============================================================
# 4-bit
# ============================================================

@torch.no_grad()
def _pack_4bit_codes(
    codes: torch.Tensor,
) -> torch.Tensor:
    """
    codes: [..., N], N even, values 0..15
    out:   [..., N/2] uint8
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError(
            "4-bit code count must be even."
        )

    codes_u8 = codes.to(torch.uint8)

    low = codes_u8[..., 0::2]
    high = codes_u8[..., 1::2]

    return (
        low |
        (high << 4)
    ).contiguous()


@torch.no_grad()
def _unpack_4bit_codes(
    packed: torch.Tensor,
) -> torch.Tensor:
    """
    packed: [..., N/2] uint8
    out:    [..., N] int64
    """
    if packed.dtype != torch.uint8:
        raise ValueError(
            f"packed must be torch.uint8, got {packed.dtype}."
        )

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    return torch.stack(
        [low, high],
        dim=-1,
    ).flatten(
        start_dim=-2
    ).to(torch.long).contiguous()


# ============================================================
# 2-bit
# ============================================================

@torch.no_grad()
def _pack_2bit_codes(
    codes: torch.Tensor,
) -> torch.Tensor:
    """
    codes: [..., N], N divisible by 4, values 0..3
    out:   [..., N/4] uint8
    """
    if codes.shape[-1] % 4 != 0:
        raise ValueError(
            "2-bit code count must be divisible by 4."
        )

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
    ).contiguous()


@torch.no_grad()
def _unpack_2bit_codes(
    packed: torch.Tensor,
) -> torch.Tensor:
    """
    packed: [..., N/4] uint8
    out:    [..., N] int64
    """
    if packed.dtype != torch.uint8:
        raise ValueError(
            f"packed must be torch.uint8, got {packed.dtype}."
        )

    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03

    return torch.stack(
        [c0, c1, c2, c3],
        dim=-1,
    ).flatten(
        start_dim=-2
    ).to(torch.long).contiguous()


# ============================================================
# 3-bit
# ============================================================

@torch.no_grad()
def _pack_3bit_codes(
    codes: torch.Tensor,
) -> torch.Tensor:
    """
    Pack 3-bit codes.

    We pack groups of 8 codes into 24 bits = 3 bytes.

    codes:
        [..., N], N divisible by 8, values 0..7

    out:
        [..., (N/8)*3] uint8
    """
    if codes.shape[-1] % 8 != 0:
        raise ValueError(
            "3-bit code count must be divisible by 8."
        )

    if torch.any(codes < 0) or torch.any(codes > 7):
        raise ValueError(
            "3-bit codes must be in [0, 7]."
        )

    group_shape = (
        *codes.shape[:-1],
        codes.shape[-1] // 8,
        8,
    )

    groups = codes.to(torch.int32).reshape(
        group_shape
    )

    c0 = groups[..., 0]
    c1 = groups[..., 1]
    c2 = groups[..., 2]
    c3 = groups[..., 3]
    c4 = groups[..., 4]
    c5 = groups[..., 5]
    c6 = groups[..., 6]
    c7 = groups[..., 7]

    packed24 = (
        c0 |
        (c1 << 3) |
        (c2 << 6) |
        (c3 << 9) |
        (c4 << 12) |
        (c5 << 15) |
        (c6 << 18) |
        (c7 << 21)
    )

    b0 = (
        packed24 & 0xFF
    ).to(torch.uint8)

    b1 = (
        (packed24 >> 8) & 0xFF
    ).to(torch.uint8)

    b2 = (
        (packed24 >> 16) & 0xFF
    ).to(torch.uint8)

    packed = torch.stack(
        [b0, b1, b2],
        dim=-1,
    ).flatten(
        start_dim=-2
    )

    return packed.contiguous()


@torch.no_grad()
def _unpack_3bit_codes(
    packed: torch.Tensor,
) -> torch.Tensor:
    """
    Unpack 3-bit codes.

    packed:
        [..., G*3] uint8

    out:
        [..., G*8] int64
    """
    if packed.dtype != torch.uint8:
        raise ValueError(
            f"packed must be torch.uint8, got {packed.dtype}."
        )

    if packed.shape[-1] % 3 != 0:
        raise ValueError(
            "3-bit packed byte count must be divisible by 3."
        )

    groups = packed.reshape(
        *packed.shape[:-1],
        packed.shape[-1] // 3,
        3,
    ).to(torch.int32)

    b0 = groups[..., 0]
    b1 = groups[..., 1]
    b2 = groups[..., 2]

    packed24 = (
        b0 |
        (b1 << 8) |
        (b2 << 16)
    )

    c0 = (packed24 >> 0) & 0x07
    c1 = (packed24 >> 3) & 0x07
    c2 = (packed24 >> 6) & 0x07
    c3 = (packed24 >> 9) & 0x07
    c4 = (packed24 >> 12) & 0x07
    c5 = (packed24 >> 15) & 0x07
    c6 = (packed24 >> 18) & 0x07
    c7 = (packed24 >> 21) & 0x07

    codes = torch.stack(
        [c0, c1, c2, c3, c4, c5, c6, c7],
        dim=-1,
    ).flatten(
        start_dim=-2
    )

    return codes.to(torch.long).contiguous()


# ============================================================
# Public Stage-1 3-bit pack/unpack
# ============================================================

@torch.no_grad()
def pack_polar_angle_codes_l4_d128_stage1_3bit(
    angle_codes: list[torch.Tensor],
) -> PackedPolarAnglesStage1ThreeBit:
    """
    Expected D=128, L=4 shapes:

        level 1: [..., 64], 4-bit codes
        level 2: [..., 32], 2-bit codes
        level 3: [..., 16], 3-bit codes
        level 4: [..., 8],  2-bit codes
    """
    if len(angle_codes) != 4:
        raise ValueError(
            f"Expected 4 angle levels, got {len(angle_codes)}."
        )

    l1, l2, l3, l4 = angle_codes

    if l1.shape[-1] != 64:
        raise ValueError(
            f"Level 1 expected 64 codes, got {l1.shape[-1]}."
        )

    if l2.shape[-1] != 32:
        raise ValueError(
            f"Level 2 expected 32 codes, got {l2.shape[-1]}."
        )

    if l3.shape[-1] != 16:
        raise ValueError(
            f"Level 3 expected 16 codes, got {l3.shape[-1]}."
        )

    if l4.shape[-1] != 8:
        raise ValueError(
            f"Level 4 expected 8 codes, got {l4.shape[-1]}."
        )

    return PackedPolarAnglesStage1ThreeBit(
        level1_4bit=_pack_4bit_codes(l1),
        level2_2bit=_pack_2bit_codes(l2),
        level3_3bit=_pack_3bit_codes(l3),
        level4_2bit=_pack_2bit_codes(l4),
    )


@torch.no_grad()
def unpack_polar_angle_codes_l4_d128_stage1_3bit(
    packed: PackedPolarAnglesStage1ThreeBit,
) -> list[torch.Tensor]:
    return [
        _unpack_4bit_codes(
            packed.level1_4bit
        ),
        _unpack_2bit_codes(
            packed.level2_2bit
        ),
        _unpack_3bit_codes(
            packed.level3_3bit
        ),
        _unpack_2bit_codes(
            packed.level4_2bit
        ),
    ]


def packed_polar_angle_stage1_3bit_storage_bytes(
    packed: PackedPolarAnglesStage1ThreeBit,
) -> int:
    return int(
        packed.level1_4bit.numel()
        * packed.level1_4bit.element_size()
        +
        packed.level2_2bit.numel()
        * packed.level2_2bit.element_size()
        +
        packed.level3_3bit.numel()
        * packed.level3_3bit.element_size()
        +
        packed.level4_2bit.numel()
        * packed.level4_2bit.element_size()
    )


def stage1_3bit_angle_bytes_per_vector_l4_d128() -> int:
    """
    32 + 8 + 6 + 2 = 48 bytes
    = 384 bits / 128 dims = 3.0 bpc.
    """
    return 48


def stage1_3bit_angle_bits_per_channel_l4_d128() -> float:
    return 3.0
