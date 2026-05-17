from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    root = Path(__file__).resolve().parent
    src = root / "csrc" / "turboquant_l2_combined_factor_lut_fp16_storage_cuda.cu"

    _EXT = load(
        name="turboquant_l2_combined_factor_lut_fp16_storage_cuda_ext",
        sources=[str(src)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
        ],
        extra_cflags=[
            "-O3",
        ],
        verbose=False,
    )
    return _EXT


def _trig(centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = centroids.contiguous().to(torch.float32)
    return torch.cos(c).contiguous(), torch.sin(c).contiguous()


@torch.no_grad()
def turboquant_polar_tree_l2_combined_lut_fp16_polar_only_cuda(
    *,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l2_factor_lut_fp16: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
) -> torch.Tensor:
    """
    Polar-only tree score using an FP16-stored L2 combined factor LUT.

    The LUT is loaded as half and converted to float32 in-register before
    downstream L3/L4 Polar tree arithmetic. This isolates the impact of
    shrinking the L2 combined LUT from 4 MB to 2 MB while preserving the
    existing compute path after the lookup.
    """
    ext = _load_ext()

    for name, tensor in {
        "packed_meta": packed_meta,
        "radii": radii,
        "l2_factor_lut_fp16": l2_factor_lut_fp16,
    }.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA.")

    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_tree_l2_combined_lut_fp16_polar_only_cuda(
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        l2_factor_lut_fp16.contiguous().to(torch.float16),
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
    )


@torch.no_grad()
def turboquant_polar_tree_l2_combined_lut_fp16_lane_nibble_fused_logits_cuda(
    *,
    q_projected: torch.Tensor,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l2_factor_lut_fp16: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> torch.Tensor:
    """
    Full fused logits with:
      - FP16-stored L2 combined factor LUT for the Polar branch
      - lane-nibble QJL sign layout

    This compares directly against the current best FP32-LUT full fused path:
      L2 combined factor LUT (FP32 storage) + lane-nibble QJL.
    """
    ext = _load_ext()

    for name, tensor in {
        "q_projected": q_projected,
        "packed_meta": packed_meta,
        "radii": radii,
        "l2_factor_lut_fp16": l2_factor_lut_fp16,
        "lane_nibble_qjl_signs": lane_nibble_qjl_signs,
        "qjl_norms": qjl_norms,
    }.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA.")

    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_tree_l2_combined_lut_fp16_lane_nibble_fused_logits_cuda(
        q_projected.contiguous().to(torch.float32),
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        l2_factor_lut_fp16.contiguous().to(torch.float16),
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        lane_nibble_qjl_signs.contiguous(),
        qjl_norms.contiguous().to(torch.float16),
    )
