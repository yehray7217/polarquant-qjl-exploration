from __future__ import annotations

import torch


@torch.no_grad()
def pack_qjl_signs_1bit(
    signs: torch.Tensor,
) -> torch.Tensor:
    """
    Pack QJL signs into 1-bit storage.

    Input:
        signs: [..., M]
            Expected values are positive / non-positive.
            M must be divisible by 8.

    Output:
        packed: [..., M/8] uint8

    Encoding:
        sign > 0 -> bit 1
        sign <=0 -> bit 0
    """
    if signs.shape[-1] % 8 != 0:
        raise ValueError(
            f"QJL sign dimension must be divisible by 8, "
            f"got M={signs.shape[-1]}."
        )

    if signs.dtype == torch.bool:
        bits = signs.to(torch.uint8)
    else:
        bits = (signs > 0).to(torch.uint8)

    b0 = bits[..., 0::8]
    b1 = bits[..., 1::8]
    b2 = bits[..., 2::8]
    b3 = bits[..., 3::8]
    b4 = bits[..., 4::8]
    b5 = bits[..., 5::8]
    b6 = bits[..., 6::8]
    b7 = bits[..., 7::8]

    packed = (
        b0 |
        (b1 << 1) |
        (b2 << 2) |
        (b3 << 3) |
        (b4 << 4) |
        (b5 << 5) |
        (b6 << 6) |
        (b7 << 7)
    )

    return packed.contiguous()


@torch.no_grad()
def unpack_qjl_signs_1bit(
    packed: torch.Tensor,
) -> torch.Tensor:
    """
    Unpack 1-bit QJL sign bits.

    Input:
        packed: [..., M/8] uint8

    Output:
        bits: [..., M] bool

            bit 1 -> True
            bit 0 -> False

    IMPORTANT:
        qjl_inner_product_estimate() uses sign-bit agreement,
        not +/-1 dense signs. Therefore this function must
        return boolean sign bits, preserving the original QJL
        estimator semantics.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(
            f"packed must be torch.uint8, got {packed.dtype}."
        )

    b0 = (packed >> 0) & 0x01
    b1 = (packed >> 1) & 0x01
    b2 = (packed >> 2) & 0x01
    b3 = (packed >> 3) & 0x01
    b4 = (packed >> 4) & 0x01
    b5 = (packed >> 5) & 0x01
    b6 = (packed >> 6) & 0x01
    b7 = (packed >> 7) & 0x01

    bits = torch.stack(
        [b0, b1, b2, b3, b4, b5, b6, b7],
        dim=-1,
    ).flatten(start_dim=-2)

    return bits.to(torch.bool).contiguous()

def packed_qjl_sign_storage_bytes(
    packed: torch.Tensor,
) -> int:
    return int(
        packed.numel() *
        packed.element_size()
    )
