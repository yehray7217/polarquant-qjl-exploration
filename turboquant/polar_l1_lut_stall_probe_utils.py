from __future__ import annotations

import torch


@torch.no_grad()
def unpack_polar_angle_codes_u8(
    *,
    packed_l1: torch.Tensor,
    packed_l2: torch.Tensor,
    packed_l3: torch.Tensor,
    packed_l4: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expand packed Polar angle codes into direct uint8 code tensors.

    Expected packed shapes for the existing 4-level PolarQuant format:
      - L1: [B,H,T,32]  -> 64 4-bit codes
      - L2: [B,H,T, 8]  -> 32 2-bit codes
      - L3: [B,H,T, 4]  -> 16 2-bit codes
      - L4: [B,H,T, 2]  ->  8 2-bit codes

    Returns:
      - l1_codes: [B,H,T,64] uint8
      - l2_codes: [B,H,T,32] uint8
      - l3_codes: [B,H,T,16] uint8
      - l4_codes: [B,H,T, 8] uint8

    These widened tensors are for profiling / stall source attribution, not a
    proposed final storage format.
    """
    tensors = {
        "packed_l1": packed_l1,
        "packed_l2": packed_l2,
        "packed_l3": packed_l3,
        "packed_l4": packed_l4,
    }
    for name, tensor in tensors.items():
        if tensor.dtype != torch.uint8:
            raise ValueError(f"{name} must be uint8, got {tensor.dtype}.")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must be [B,H,T,bytes], got {tuple(tensor.shape)}.")

    expected_last_dims = {
        "packed_l1": 32,
        "packed_l2": 8,
        "packed_l3": 4,
        "packed_l4": 2,
    }
    for name, expected in expected_last_dims.items():
        got = int(tensors[name].shape[-1])
        if got != expected:
            raise ValueError(f"{name} last dim must be {expected}, got {got}.")

    base_shape = tuple(int(x) for x in packed_l1.shape[:-1])
    for name, tensor in tensors.items():
        if tuple(int(x) for x in tensor.shape[:-1]) != base_shape:
            raise ValueError(f"{name} prefix shape mismatch: {tuple(tensor.shape[:-1])} vs {base_shape}.")

    def unpack(
        packed: torch.Tensor,
        *,
        codes_per_byte: int,
        bits_per_code: int,
        num_codes: int,
    ) -> torch.Tensor:
        out = []
        mask = (1 << bits_per_code) - 1
        for code_idx in range(num_codes):
            byte_idx = code_idx // codes_per_byte
            shift = (code_idx % codes_per_byte) * bits_per_code
            out.append(((packed[..., byte_idx] >> shift) & mask).to(torch.uint8))
        return torch.stack(out, dim=-1).contiguous()

    l1 = unpack(
        packed_l1,
        codes_per_byte=2,
        bits_per_code=4,
        num_codes=64,
    )
    l2 = unpack(
        packed_l2,
        codes_per_byte=4,
        bits_per_code=2,
        num_codes=32,
    )
    l3 = unpack(
        packed_l3,
        codes_per_byte=4,
        bits_per_code=2,
        num_codes=16,
    )
    l4 = unpack(
        packed_l4,
        codes_per_byte=4,
        bits_per_code=2,
        num_codes=8,
    )
    return l1, l2, l3, l4
