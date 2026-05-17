from __future__ import annotations

import torch


PACKED_META32_BYTES = 32

L1_OFFSET = 0
L1_BYTES = 16

L2_OFFSET = 16
L2_BYTES = 4

L3_OFFSET = 20
L3_BYTES = 2

L4_OFFSET = 22
L4_BYTES = 1

PADDING_OFFSET = 23
PADDING_BYTES = 1

QJL_OFFSET = 24
QJL_BYTES = 8


@torch.no_grad()
def build_polarquant_3bpc_packed_meta32_blob(
    *,
    packed_l1_2bit: torch.Tensor,
    packed_l2_1bit: torch.Tensor,
    packed_l3_1bit: torch.Tensor,
    packed_l4_1bit: torch.Tensor,
    packed_qjl_signs: torch.Tensor,
) -> torch.Tensor:
    """
    Build contiguous aligned metadata:
        [B,H,T,32] uint8

    Layout:
        0  ~ 15 : L1 2-bit polar codes      [16 B]
        16 ~ 19 : L2 1-bit polar codes       [4 B]
        20 ~ 21 : L3 1-bit polar codes       [2 B]
        22      : L4 1-bit polar codes       [1 B]
        23      : zero padding               [1 B]
        24 ~ 31 : QJL M=64 sign bits         [8 B]

    Physical K accounting for the fused logits path:
        meta32 32 B + radii 16 B + qjl_norm 2 B = 50 B / D=128
        => 3.125 bits/channel.
    """
    tensors = {
        "packed_l1_2bit": packed_l1_2bit,
        "packed_l2_1bit": packed_l2_1bit,
        "packed_l3_1bit": packed_l3_1bit,
        "packed_l4_1bit": packed_l4_1bit,
        "packed_qjl_signs": packed_qjl_signs,
    }

    for name, tensor in tensors.items():
        if tensor.dtype != torch.uint8:
            raise ValueError(f"{name} must be torch.uint8, got {tensor.dtype}.")
        if tensor.dim() != 4:
            raise ValueError(
                f"{name} must be rank-4 [B,H,T,C], got {tuple(tensor.shape)}."
            )

    B, H, T, _ = packed_l1_2bit.shape
    expected = {
        "packed_l1_2bit": (B, H, T, L1_BYTES),
        "packed_l2_1bit": (B, H, T, L2_BYTES),
        "packed_l3_1bit": (B, H, T, L3_BYTES),
        "packed_l4_1bit": (B, H, T, L4_BYTES),
        "packed_qjl_signs": (B, H, T, QJL_BYTES),
    }

    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(
                f"{name} shape mismatch: expected {shape}, got {tuple(tensors[name].shape)}."
            )

    device = packed_l1_2bit.device
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(
                f"{name} device mismatch: expected {device}, got {tensor.device}."
            )

    meta = torch.zeros(
        (B, H, T, PACKED_META32_BYTES),
        device=device,
        dtype=torch.uint8,
    )

    meta[..., L1_OFFSET:L1_OFFSET + L1_BYTES] = packed_l1_2bit
    meta[..., L2_OFFSET:L2_OFFSET + L2_BYTES] = packed_l2_1bit
    meta[..., L3_OFFSET:L3_OFFSET + L3_BYTES] = packed_l3_1bit
    meta[..., L4_OFFSET:L4_OFFSET + L4_BYTES] = packed_l4_1bit
    meta[..., QJL_OFFSET:QJL_OFFSET + QJL_BYTES] = packed_qjl_signs

    return meta.contiguous()


def packed_meta32_storage_bytes(packed_meta32: torch.Tensor) -> int:
    return int(
        packed_meta32.numel()
        * packed_meta32.element_size()
    )
