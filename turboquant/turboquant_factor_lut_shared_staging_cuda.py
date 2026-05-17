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
    src = root / "csrc" / "turboquant_factor_lut_shared_staging_cuda.cu"

    _EXT = load(
        name="turboquant_factor_lut_shared_staging_cuda_ext",
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
def turboquant_polar_tree_l1_lut_shared_polar_only_cuda(
    *,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l1_factor_lut: torch.Tensor,
    centroids_l2: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
) -> torch.Tensor:
    """
    Polar-only L1-LUT score with per-CTA shared-memory staging of the
    head-local factor LUT [64,16] = 4 KB.

    Compared with the existing polar-only L1-LUT kernel, Polar math and
    packed meta layout are unchanged; only factor LUT access changes:
      global scattered lookup -> CTA-coalesced global staging + shared lookup.
    """
    ext = _load_ext()

    for name, tensor in {
        "packed_meta": packed_meta,
        "radii": radii,
        "l1_factor_lut": l1_factor_lut,
    }.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA.")

    cos_l2, sin_l2 = _trig(centroids_l2)
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_tree_l1_lut_shared_polar_only_cuda(
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        l1_factor_lut.contiguous().to(torch.float32),
        cos_l2,
        sin_l2,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
    )


@torch.no_grad()
def turboquant_polar_tree_l1_lut_shared_lane_nibble_fused_logits_cuda(
    *,
    q_projected: torch.Tensor,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l1_factor_lut: torch.Tensor,
    centroids_l2: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> torch.Tensor:
    """
    Current best full fused path plus per-CTA shared-memory staging of the
    head-local L1 factor LUT.

    Preserves:
      - 5.125-bpc Polar metadata path
      - L1 factor LUT math
      - lane-nibble QJL sign layout
      - exact fused logits semantics

    Changes only:
      factor LUT access:
        global scattered lookup -> CTA-coalesced global staging + shared lookup.
    """
    ext = _load_ext()

    for name, tensor in {
        "q_projected": q_projected,
        "packed_meta": packed_meta,
        "radii": radii,
        "l1_factor_lut": l1_factor_lut,
        "lane_nibble_qjl_signs": lane_nibble_qjl_signs,
        "qjl_norms": qjl_norms,
    }.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA.")

    cos_l2, sin_l2 = _trig(centroids_l2)
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_tree_l1_lut_shared_lane_nibble_fused_logits_cuda(
        q_projected.contiguous().to(torch.float32),
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        l1_factor_lut.contiguous().to(torch.float32),
        cos_l2,
        sin_l2,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        lane_nibble_qjl_signs.contiguous(),
        qjl_norms.contiguous().to(torch.float16),
    )
