from __future__ import annotations

import torch


@torch.no_grad()
def build_predecoded_l2_codes_exact(
    packed_l2_2bit: torch.Tensor,
) -> torch.Tensor:
    """
    Expand packed L2 2-bit codes into exact uint8 codes.

    Input:
      packed_l2_2bit: [B,H,T,8] uint8
        Each byte stores four 2-bit L2 codes.

    Output:
      predecoded_l2_codes: [B,H,T,32] uint8
        Code order matches l2_idx in the CUDA kernel:
          byte_idx = l2_idx >> 2
          shift    = (l2_idx & 3) * 2
    """
    if not packed_l2_2bit.is_cuda:
        raise ValueError("packed_l2_2bit must be CUDA.")
    if packed_l2_2bit.dtype != torch.uint8:
        raise ValueError("packed_l2_2bit must be uint8.")
    if packed_l2_2bit.ndim != 4:
        raise ValueError("packed_l2_2bit must be [B,H,T,8].")
    if packed_l2_2bit.shape[-1] != 8:
        raise ValueError("packed_l2_2bit last dim must be 8.")

    x = packed_l2_2bit.contiguous()
    shifts = torch.tensor(
        [0, 2, 4, 6],
        device=x.device,
        dtype=torch.uint8,
    ).view(1, 1, 1, 1, 4)

    bytes_expanded = x.unsqueeze(-1)
    codes = torch.bitwise_and(
        torch.bitwise_right_shift(bytes_expanded, shifts),
        torch.tensor(0x03, device=x.device, dtype=torch.uint8),
    )

    return codes.reshape(*x.shape[:-1], 32).contiguous()
