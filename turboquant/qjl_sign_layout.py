from __future__ import annotations

import torch


@torch.no_grad()
def build_qjl_lane_nibble_signs(
    packed_qjl_signs: torch.Tensor,
) -> torch.Tensor:
    """
    Repack QJL M=128 sign bits from sketch-major bytes to lane-major nibbles.

    Input:
        packed_qjl_signs: [B,H,T,16] uint8
            Original ordering. Bit `j` represents sign at sketch index j.

    Output:
        lane_nibble_signs: [B,H,T,16] uint8
            Each key still uses 16 bytes. Byte i packs:
              - low nibble:  lane 2*i signs for indices lane + 32*k, k=0..3
              - high nibble: lane 2*i+1 signs for indices lane + 32*k, k=0..3

    This layout lets CUDA lane `lane` load its four sign bits from one nibble.
    It preserves the 16-byte physical QJL sign storage.
    """
    if packed_qjl_signs.dtype != torch.uint8:
        raise ValueError(
            f"packed_qjl_signs must be torch.uint8, got {packed_qjl_signs.dtype}."
        )
    if packed_qjl_signs.ndim != 4:
        raise ValueError(
            "packed_qjl_signs must be [B,H,T,16], "
            f"got {tuple(packed_qjl_signs.shape)}."
        )
    if int(packed_qjl_signs.shape[-1]) != 16:
        raise ValueError(
            "packed_qjl_signs last dimension must be 16 bytes for M=128, "
            f"got {packed_qjl_signs.shape[-1]}."
        )

    device = packed_qjl_signs.device
    sketch_idx = torch.arange(128, device=device, dtype=torch.long)
    byte_idx = sketch_idx // 8
    bit_idx = sketch_idx % 8

    sketch_bits = (
        (
            packed_qjl_signs[..., byte_idx]
            >> bit_idx.view(*(1 for _ in range(packed_qjl_signs.ndim - 1)), 128)
        )
        & 0x01
    ).to(torch.uint8)

    # [B,H,T,4,32] -> [B,H,T,32,4]
    lane_bits = sketch_bits.reshape(*packed_qjl_signs.shape[:-1], 4, 32)
    lane_bits = lane_bits.permute(0, 1, 2, 4, 3).contiguous()

    lane_nibbles = (
        lane_bits[..., 0]
        | (lane_bits[..., 1] << 1)
        | (lane_bits[..., 2] << 2)
        | (lane_bits[..., 3] << 3)
    ).to(torch.uint8)

    low = lane_nibbles[..., 0::2]
    high = lane_nibbles[..., 1::2] << 4
    return (low | high).contiguous()
