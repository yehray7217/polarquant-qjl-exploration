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
    src = root / "csrc" / "turboquant_logits_tree_l1_lut_ablation_cuda.cu"

    _EXT = load(
        name="turboquant_polar_tree_l1_lut_ablation_cuda_ext",
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
def turboquant_polar_tree_l1_lut_polar_only_cuda(
    *,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l1_factor_lut: torch.Tensor,
    centroids_l2: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
) -> torch.Tensor:
    """
    Polar-only ablation for the 5.125-bpc L1-LUT tree path.

    Returns:
        [1,H,1,T] float32 Polar tree contribution only.
    """
    ext = _load_ext()

    if not packed_meta.is_cuda:
        raise ValueError("packed_meta must be CUDA.")
    if not radii.is_cuda:
        raise ValueError("radii must be CUDA.")
    if not l1_factor_lut.is_cuda:
        raise ValueError("l1_factor_lut must be CUDA.")

    cos_l2, sin_l2 = _trig(centroids_l2)
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_tree_l1_lut_polar_only_cuda(
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
def turboquant_polar_tree_l1_lut_qjl_only_cuda(
    *,
    q_projected: torch.Tensor,
    packed_meta: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> torch.Tensor:
    """
    QJL-only ablation for the 5.125-bpc L1-LUT tree path.

    Returns:
        [1,H,1,T] float32 QJL residual correction only.
    """
    ext = _load_ext()

    if not q_projected.is_cuda:
        raise ValueError("q_projected must be CUDA.")
    if not packed_meta.is_cuda:
        raise ValueError("packed_meta must be CUDA.")
    if not qjl_norms.is_cuda:
        raise ValueError("qjl_norms must be CUDA.")

    return ext.turboquant_polar_tree_l1_lut_qjl_only_cuda(
        q_projected.contiguous().to(torch.float32),
        packed_meta.contiguous(),
        qjl_norms.contiguous().to(torch.float16),
    )
