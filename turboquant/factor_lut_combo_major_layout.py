from __future__ import annotations

import torch


@torch.no_grad()
def build_l2_factor_lut_combo_major_fp16(
    l2_factor_lut_fp16: torch.Tensor,
) -> torch.Tensor:
    """
    Convert factor LUT layout:
      input : [H, 32 groups, 1024 combos]
      output: [H, 1024 combos, 32 groups]

    Values remain exact. This is a memory-layout ablation only.
    """
    if not l2_factor_lut_fp16.is_cuda:
        raise ValueError("l2_factor_lut_fp16 must be CUDA.")
    if l2_factor_lut_fp16.dtype != torch.float16:
        raise ValueError("l2_factor_lut_fp16 must be float16.")
    if l2_factor_lut_fp16.ndim != 3:
        raise ValueError("l2_factor_lut_fp16 must be [H,32,1024].")
    if l2_factor_lut_fp16.shape[1] != 32 or l2_factor_lut_fp16.shape[2] != 1024:
        raise ValueError("Expected l2_factor_lut_fp16 shape [H,32,1024].")

    return l2_factor_lut_fp16.permute(0, 2, 1).contiguous()


@torch.no_grad()
def estimate_combo_major_lut_locality(
    *,
    packed_l1_4bit: torch.Tensor,
    packed_l2_2bit: torch.Tensor,
    sample_tokens: int = 4096,
) -> dict[str, float | int]:
    """
    Estimate warp-level LUT address locality for combo-major layout.

    We sample the first `sample_tokens` tokens per head and reconstruct the
    32 combo indices used by one warp/token:
      combo = (c2 << 8) | (c1b << 4) | c1a

    For combo-major [combo, group] layout, each lookup is indexed by:
      flat = combo * 32 + group

    A 128B sector contains 64 FP16 entries. We estimate the number of unique
    128B sectors touched per warp for the factor-LUT lookup.
    """
    if packed_l1_4bit.dtype != torch.uint8 or packed_l2_2bit.dtype != torch.uint8:
        raise ValueError("packed_l1_4bit and packed_l2_2bit must be uint8.")
    if packed_l1_4bit.ndim != 4 or packed_l2_2bit.ndim != 4:
        raise ValueError("Packed code tensors must be [B,H,T,*].")
    if packed_l1_4bit.shape[-1] != 32:
        raise ValueError("packed_l1_4bit last dim must be 32 bytes.")
    if packed_l2_2bit.shape[-1] != 8:
        raise ValueError("packed_l2_2bit last dim must be 8 bytes.")

    t = min(int(sample_tokens), int(packed_l1_4bit.shape[2]))
    if t <= 0:
        return {
            "sample_tokens": 0,
            "sample_warps": 0,
            "mean_unique_combo_per_warp": float("nan"),
            "mean_combo_major_128b_sectors_per_warp": float("nan"),
            "baseline_group_major_128b_sectors_per_warp": 32.0,
        }

    l1 = packed_l1_4bit[:, :, :t, :].contiguous()
    l2 = packed_l2_2bit[:, :, :t, :].contiguous()

    # Decode 64 L1 4-bit codes -> [B,H,T,64].
    low = torch.bitwise_and(l1, torch.tensor(0x0F, device=l1.device, dtype=torch.uint8))
    high = torch.bitwise_and(
        torch.bitwise_right_shift(l1, torch.tensor(4, device=l1.device, dtype=torch.uint8)),
        torch.tensor(0x0F, device=l1.device, dtype=torch.uint8),
    )
    l1_codes = torch.stack([low, high], dim=-1).reshape(*l1.shape[:-1], 64)

    # Decode 32 L2 2-bit codes -> [B,H,T,32].
    shifts = torch.tensor([0, 2, 4, 6], device=l2.device, dtype=torch.uint8).view(1, 1, 1, 1, 4)
    l2_codes = torch.bitwise_and(
        torch.bitwise_right_shift(l2.unsqueeze(-1), shifts),
        torch.tensor(0x03, device=l2.device, dtype=torch.uint8),
    ).reshape(*l2.shape[:-1], 32)

    # For l2_idx = 0..31, consume L1 pair 2*i and 2*i+1.
    c1a = l1_codes[..., 0::2]
    c1b = l1_codes[..., 1::2]
    c2 = l2_codes

    combo = (
        c2.to(torch.int32) * 256
        + c1b.to(torch.int32) * 16
        + c1a.to(torch.int32)
    )  # [B,H,T,32]

    # Unique combos per warp/token.
    combo_sorted, _ = torch.sort(combo, dim=-1)
    combo_changes = (combo_sorted[..., 1:] != combo_sorted[..., :-1]).to(torch.int32)
    unique_combo_count = 1 + combo_changes.sum(dim=-1)

    groups = torch.arange(32, device=combo.device, dtype=torch.int32).view(1, 1, 1, 32)
    combo_major_flat = combo * 32 + groups
    combo_major_sector = torch.div(combo_major_flat, 64, rounding_mode="floor")
    sector_sorted, _ = torch.sort(combo_major_sector, dim=-1)
    sector_changes = (sector_sorted[..., 1:] != sector_sorted[..., :-1]).to(torch.int32)
    unique_sector_count = 1 + sector_changes.sum(dim=-1)

    sample_warps = int(unique_combo_count.numel())

    return {
        "sample_tokens": int(t),
        "sample_warps": sample_warps,
        "mean_unique_combo_per_warp": float(unique_combo_count.float().mean().item()),
        "mean_combo_major_128b_sectors_per_warp": float(unique_sector_count.float().mean().item()),
        "baseline_group_major_128b_sectors_per_warp": 32.0,
        "mean_sector_reduction_vs_group_major": float(
            32.0 - unique_sector_count.float().mean().item()
        ),
    }
