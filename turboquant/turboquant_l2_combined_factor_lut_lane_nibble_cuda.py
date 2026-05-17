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
    src = root / "csrc" / "turboquant_l2_combined_factor_lut_lane_nibble_cuda.cu"

    _EXT = load(
        name="turboquant_l2_combined_factor_lut_lane_nibble_cuda_ext",
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
def turboquant_polar_tree_l2_combined_lut_polar_only_cuda(
    *,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l2_factor_lut: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
) -> torch.Tensor:
    """
    Polar-only tree score using the L2 combined factor LUT.

    This is the "on-demand lookup reduction" path:
      - previous L1 path:
          2 x L1 factor LUT loads + L2 arithmetic combine
      - this L2 path:
          1 x combined L2 LUT load indexed by (c1a, c1b, c2)

    Output:
        [1,H,1,T] float32 Polar-only contribution.
    """
    ext = _load_ext()

    for name, tensor in {
        "packed_meta": packed_meta,
        "radii": radii,
        "l2_factor_lut": l2_factor_lut,
    }.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA.")

    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_tree_l2_combined_lut_polar_only_cuda(
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        l2_factor_lut.contiguous().to(torch.float32),
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
    )


@torch.no_grad()
def turboquant_polar_tree_l2_combined_lut_lane_nibble_fused_logits_cuda(
    *,
    q_projected: torch.Tensor,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l2_factor_lut: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> torch.Tensor:
    """
    Full fused logits with:
      - L2 combined factor LUT for Polar score
      - lane-nibble QJL sign layout

    This directly compares against the current best:
      L1 factor LUT global lookup + lane-nibble QJL.
    """
    ext = _load_ext()

    for name, tensor in {
        "q_projected": q_projected,
        "packed_meta": packed_meta,
        "radii": radii,
        "l2_factor_lut": l2_factor_lut,
        "lane_nibble_qjl_signs": lane_nibble_qjl_signs,
        "qjl_norms": qjl_norms,
    }.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA.")

    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_tree_l2_combined_lut_lane_nibble_fused_logits_cuda(
        q_projected.contiguous().to(torch.float32),
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        l2_factor_lut.contiguous().to(torch.float32),
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        lane_nibble_qjl_signs.contiguous(),
        qjl_norms.contiguous().to(torch.float16),
    )
