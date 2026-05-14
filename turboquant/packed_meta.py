from __future__ import annotations

import torch


PACKED_META_BYTES = 64

L1_OFFSET = 0
L1_BYTES = 32

L2_OFFSET = 32
L2_BYTES = 8

L3_OFFSET = 40
L3_BYTES = 4

L4_OFFSET = 44
L4_BYTES = 2

PADDING_OFFSET = 46
PADDING_BYTES = 2

QJL_OFFSET = 48
QJL_BYTES = 16


@torch.no_grad()
def build_turboquant_packed_meta_blob(
    *,
    packed_l1: torch.Tensor,
    packed_l2: torch.Tensor,
    packed_l3: torch.Tensor,
    packed_l4: torch.Tensor,
    packed_qjl_signs: torch.Tensor,
) -> torch.Tensor:
    """
    Build contiguous packed metadata blob:

        [B,H,T,64] uint8

    Layout:
        0  ~ 31 : packed_l1       [32 B]
        32 ~ 39 : packed_l2       [8 B]
        40 ~ 43 : packed_l3       [4 B]
        44 ~ 45 : packed_l4       [2 B]
        46 ~ 47 : zero padding    [2 B]
        48 ~ 63 : qjl signs       [16 B]
    """
    tensors = {
        "packed_l1": packed_l1,
        "packed_l2": packed_l2,
        "packed_l3": packed_l3,
        "packed_l4": packed_l4,
        "packed_qjl_signs": packed_qjl_signs,
    }

    for name, tensor in tensors.items():
        if tensor.dtype != torch.uint8:
            raise ValueError(
                f"{name} must be torch.uint8, got {tensor.dtype}."
            )

        if tensor.dim() != 4:
            raise ValueError(
                f"{name} must be rank-4 [B,H,T,C], got shape={tuple(tensor.shape)}."
            )

    B, H, T, c1 = packed_l1.shape

    expected_shapes = {
        "packed_l1": (B, H, T, L1_BYTES),
        "packed_l2": (B, H, T, L2_BYTES),
        "packed_l3": (B, H, T, L3_BYTES),
        "packed_l4": (B, H, T, L4_BYTES),
        "packed_qjl_signs": (B, H, T, QJL_BYTES),
    }

    for name, expected_shape in expected_shapes.items():
        got_shape = tuple(tensors[name].shape)
        if got_shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, got {got_shape}."
            )

    device = packed_l1.device

    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(
                f"{name} device mismatch: expected {device}, got {tensor.device}."
            )

    meta = torch.zeros(
        (B, H, T, PACKED_META_BYTES),
        device=device,
        dtype=torch.uint8,
    )

    meta[..., L1_OFFSET:L1_OFFSET + L1_BYTES] = packed_l1
    meta[..., L2_OFFSET:L2_OFFSET + L2_BYTES] = packed_l2
    meta[..., L3_OFFSET:L3_OFFSET + L3_BYTES] = packed_l3
    meta[..., L4_OFFSET:L4_OFFSET + L4_BYTES] = packed_l4
    meta[..., QJL_OFFSET:QJL_OFFSET + QJL_BYTES] = packed_qjl_signs

    return meta.contiguous()


@torch.no_grad()
def unpack_turboquant_packed_meta_blob(
    packed_meta: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Debug/helper unpack for verification.
    """
    if packed_meta.dtype != torch.uint8:
        raise ValueError(
            f"packed_meta must be torch.uint8, got {packed_meta.dtype}."
        )

    if packed_meta.dim() != 4:
        raise ValueError(
            f"packed_meta must be [B,H,T,64], got shape={tuple(packed_meta.shape)}."
        )

    if packed_meta.shape[-1] != PACKED_META_BYTES:
        raise ValueError(
            f"packed_meta last dim must be {PACKED_META_BYTES}, "
            f"got {packed_meta.shape[-1]}."
        )

    return {
        "packed_l1": packed_meta[..., L1_OFFSET:L1_OFFSET + L1_BYTES].contiguous(),
        "packed_l2": packed_meta[..., L2_OFFSET:L2_OFFSET + L2_BYTES].contiguous(),
        "packed_l3": packed_meta[..., L3_OFFSET:L3_OFFSET + L3_BYTES].contiguous(),
        "packed_l4": packed_meta[..., L4_OFFSET:L4_OFFSET + L4_BYTES].contiguous(),
        "padding": packed_meta[..., PADDING_OFFSET:PADDING_OFFSET + PADDING_BYTES].contiguous(),
        "packed_qjl_signs": packed_meta[..., QJL_OFFSET:QJL_OFFSET + QJL_BYTES].contiguous(),
    }


def packed_meta_storage_bytes(
    packed_meta: torch.Tensor,
) -> int:
    return int(
        packed_meta.numel()
        * packed_meta.element_size()
    )
