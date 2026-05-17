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
    src = root / "csrc" / "turboquant_logits_tree_lut_cuda.cu"

    _EXT = load(
        name="turboquant_polar_tree_lut_cuda_ext",
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


def _common_inputs(
    *,
    q_projected: torch.Tensor,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not q_projected.is_cuda:
        raise ValueError("q_projected must be CUDA.")
    if not packed_meta.is_cuda:
        raise ValueError("packed_meta must be CUDA.")
    if not radii.is_cuda:
        raise ValueError("radii must be CUDA.")
    if not qjl_norms.is_cuda:
        raise ValueError("qjl_norms must be CUDA.")

    return (
        q_projected.contiguous().to(torch.float32),
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        qjl_norms.contiguous().to(torch.float16),
    )


def _trig(centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = centroids.contiguous().to(torch.float32)
    return torch.cos(c).contiguous(), torch.sin(c).contiguous()


@torch.no_grad()
def turboquant_polar_tree_l1_lut_fused_logits_cuda(
    *,
    q_projected: torch.Tensor,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l1_factor_lut: torch.Tensor,
    centroids_l2: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> torch.Tensor:
    """
    5.125-bpc Polar tree scoring with a query-side L1 factor LUT.

    The compressed K format stays unchanged:
        meta64, M=128, bits=(4,2,2,2)
    """
    ext = _load_ext()
    qproj, meta, radii_f16, norms_f16 = _common_inputs(
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=radii,
        qjl_norms=qjl_norms,
    )

    if not l1_factor_lut.is_cuda:
        raise ValueError("l1_factor_lut must be CUDA.")
    lut = l1_factor_lut.contiguous().to(torch.float32)

    cos_l2, sin_l2 = _trig(centroids_l2)
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_tree_l1_lut_fused_logits_cuda(
        qproj,
        meta,
        radii_f16,
        lut,
        cos_l2,
        sin_l2,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        norms_f16,
    )


@torch.no_grad()
def turboquant_polar_tree_l2_lut_fused_logits_cuda(
    *,
    q_projected: torch.Tensor,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l2_factor_lut: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> torch.Tensor:
    """
    5.125-bpc Polar tree scoring with a query-side L2 factor LUT.

    The compressed K format stays unchanged:
        meta64, M=128, bits=(4,2,2,2)
    """
    ext = _load_ext()
    qproj, meta, radii_f16, norms_f16 = _common_inputs(
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=radii,
        qjl_norms=qjl_norms,
    )

    if not l2_factor_lut.is_cuda:
        raise ValueError("l2_factor_lut must be CUDA.")
    lut = l2_factor_lut.contiguous().to(torch.float32)

    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_tree_l2_lut_fused_logits_cuda(
        qproj,
        meta,
        radii_f16,
        lut,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        norms_f16,
    )
