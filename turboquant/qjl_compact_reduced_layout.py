from __future__ import annotations
import torch

@torch.no_grad()
def pad_qjl_signs_to_meta64_16bytes(packed_qjl_signs: torch.Tensor) -> torch.Tensor:
    """Pad M=32/64 packed signs to 16 bytes only for packed_meta builder compatibility.
    Current Polar tree kernels read polar words from packed_meta; QJL signs are passed separately.
    """
    if packed_qjl_signs.dtype != torch.uint8 or packed_qjl_signs.ndim != 4:
        raise ValueError('packed_qjl_signs must be [B,H,T,bytes] uint8')
    n = int(packed_qjl_signs.shape[-1])
    if n > 16:
        raise ValueError('packed_qjl_signs last dim must be <=16')
    if n == 16:
        return packed_qjl_signs.contiguous()
    out = torch.zeros(*packed_qjl_signs.shape[:-1], 16, device=packed_qjl_signs.device, dtype=torch.uint8)
    out[..., :n] = packed_qjl_signs
    return out.contiguous()

@torch.no_grad()
def build_qjl_compact_signs_m64(packed_qjl_signs_m64: torch.Tensor) -> torch.Tensor:
    """Repack M=64 sketch-major sign bits [B,H,T,8] -> compact lane-group [B,H,T,8].
    Each output byte packs four lanes; each lane stores two bits for indices lane and lane+32.
    """
    if packed_qjl_signs_m64.dtype != torch.uint8 or packed_qjl_signs_m64.ndim != 4 or int(packed_qjl_signs_m64.shape[-1]) != 8:
        raise ValueError('packed_qjl_signs_m64 must be [B,H,T,8] uint8')
    x = packed_qjl_signs_m64.contiguous()
    idx = torch.arange(64, device=x.device, dtype=torch.long)
    byte_idx = idx // 8
    bit_idx = idx % 8
    bits = ((x[..., byte_idx] >> bit_idx.view(*(1 for _ in range(x.ndim-1)), 64)) & 0x01).to(torch.uint8)
    lane_bits = bits.reshape(*x.shape[:-1], 2, 32).permute(0,1,2,4,3).contiguous()  # [...,32,2]
    lane_two = (lane_bits[..., 0] | (lane_bits[..., 1] << 1)).to(torch.uint8)
    packed = (
        lane_two[..., 0::4]
        | (lane_two[..., 1::4] << 2)
        | (lane_two[..., 2::4] << 4)
        | (lane_two[..., 3::4] << 6)
    )
    return packed.contiguous()

@torch.no_grad()
def build_qjl_compact_signs_m32(packed_qjl_signs_m32: torch.Tensor) -> torch.Tensor:
    """M=32 sign order already matches lane bit order. Keep [B,H,T,4] contiguous."""
    if packed_qjl_signs_m32.dtype != torch.uint8 or packed_qjl_signs_m32.ndim != 4 or int(packed_qjl_signs_m32.shape[-1]) != 4:
        raise ValueError('packed_qjl_signs_m32 must be [B,H,T,4] uint8')
    return packed_qjl_signs_m32.contiguous()
