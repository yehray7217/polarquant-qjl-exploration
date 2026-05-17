from __future__ import annotations

import torch

L2_GROUPS = 32
L2_CODES = 1024


@torch.no_grad()
def _decode_combo_chunk(
    packed_l1_4bit: torch.Tensor,
    packed_l2_2bit: torch.Tensor,
) -> torch.Tensor:
    """Return combo indices [H,T,32] for a B=1 chunk."""
    if packed_l1_4bit.shape[0] != 1 or packed_l2_2bit.shape[0] != 1:
        raise ValueError("hot-combo analysis requires B=1.")
    if packed_l1_4bit.dtype != torch.uint8 or packed_l2_2bit.dtype != torch.uint8:
        raise ValueError("packed L1/L2 tensors must be uint8.")

    l1 = packed_l1_4bit.contiguous()
    l2 = packed_l2_2bit.contiguous()

    low = torch.bitwise_and(l1, torch.tensor(0x0F, device=l1.device, dtype=torch.uint8))
    high = torch.bitwise_and(
        torch.bitwise_right_shift(l1, torch.tensor(4, device=l1.device, dtype=torch.uint8)),
        torch.tensor(0x0F, device=l1.device, dtype=torch.uint8),
    )
    l1_codes = torch.stack([low, high], dim=-1).reshape(1, l1.shape[1], l1.shape[2], 64)

    shifts = torch.tensor([0, 2, 4, 6], device=l2.device, dtype=torch.uint8).view(1, 1, 1, 1, 4)
    l2_codes = torch.bitwise_and(
        torch.bitwise_right_shift(l2.unsqueeze(-1), shifts),
        torch.tensor(0x03, device=l2.device, dtype=torch.uint8),
    ).reshape(1, l2.shape[1], l2.shape[2], 32)

    c1a = l1_codes[..., 0::2]
    c1b = l1_codes[..., 1::2]
    c2 = l2_codes
    combo = (
        c2.to(torch.int64) * 256
        + c1b.to(torch.int64) * 16
        + c1a.to(torch.int64)
    )
    return combo.squeeze(0).contiguous()


@torch.no_grad()
def build_combo_histogram(
    *,
    packed_l1_4bit: torch.Tensor,
    packed_l2_2bit: torch.Tensor,
    chunk_tokens: int = 4096,
) -> torch.Tensor:
    """Return exact combo counts [H,32,1024]."""
    if packed_l1_4bit.ndim != 4 or packed_l2_2bit.ndim != 4:
        raise ValueError("packed L1/L2 tensors must be [B,H,T,*].")
    if packed_l1_4bit.shape[:3] != packed_l2_2bit.shape[:3]:
        raise ValueError("packed L1/L2 leading dims must match.")
    if packed_l1_4bit.shape[-1] != 32:
        raise ValueError("packed_l1_4bit last dim must be 32.")
    if packed_l2_2bit.shape[-1] != 8:
        raise ValueError("packed_l2_2bit last dim must be 8.")

    H = int(packed_l1_4bit.shape[1])
    T = int(packed_l1_4bit.shape[2])
    device = packed_l1_4bit.device
    counts_flat = torch.zeros(H * L2_GROUPS * L2_CODES, dtype=torch.int64, device=device)

    group_ids = torch.arange(L2_GROUPS, device=device, dtype=torch.int64).view(1, 1, L2_GROUPS)
    head_ids = torch.arange(H, device=device, dtype=torch.int64).view(H, 1, 1)
    chunk_tokens = max(1, int(chunk_tokens))

    for start in range(0, T, chunk_tokens):
        end = min(T, start + chunk_tokens)
        combo = _decode_combo_chunk(
            packed_l1_4bit[:, :, start:end, :],
            packed_l2_2bit[:, :, start:end, :],
        )
        flat_idx = (((head_ids * L2_GROUPS + group_ids) * L2_CODES) + combo).reshape(-1)
        counts_flat += torch.bincount(flat_idx, minlength=counts_flat.numel())

    return counts_flat.view(H, L2_GROUPS, L2_CODES).contiguous()


@torch.no_grad()
def summarize_hot_combo_coverage(
    counts: torch.Tensor,
    top_ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> dict[str, float]:
    if counts.ndim != 3 or counts.shape[1:] != (L2_GROUPS, L2_CODES):
        raise ValueError("counts must be [H,32,1024].")
    totals = counts.sum(dim=-1).clamp_min(1)
    sorted_counts, _ = torch.sort(counts, dim=-1, descending=True)
    out: dict[str, float] = {}
    flat_total = counts.sum().clamp_min(1)
    for k in top_ks:
        kk = min(int(k), L2_CODES)
        cov = sorted_counts[..., :kk].sum(dim=-1).float() / totals.float()
        out[f"top{k}_mean_group_coverage"] = float(cov.mean().item())
        out[f"top{k}_min_group_coverage"] = float(cov.min().item())
        out[f"top{k}_max_group_coverage"] = float(cov.max().item())
        out[f"top{k}_global_coverage"] = float(
            sorted_counts[..., :kk].sum().float().div(flat_total.float()).item()
        )
    return out


@torch.no_grad()
def build_hot_combo_tables(
    *,
    counts: torch.Tensor,
    l2_factor_lut_fp16: torch.Tensor,
    hot_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if l2_factor_lut_fp16.dtype != torch.float16:
        raise ValueError("l2_factor_lut_fp16 must be float16.")
    if l2_factor_lut_fp16.shape != counts.shape:
        raise ValueError("counts and l2_factor_lut_fp16 must have identical [H,32,1024] shape.")
    hot_k = int(hot_k)
    _, top_idx = torch.topk(counts, k=hot_k, dim=-1, largest=True, sorted=True)
    hot_ids = top_idx.to(torch.int16).contiguous()
    hot_vals = torch.gather(
        l2_factor_lut_fp16,
        dim=-1,
        index=top_idx.to(torch.int64),
    ).contiguous()
    return hot_ids, hot_vals
