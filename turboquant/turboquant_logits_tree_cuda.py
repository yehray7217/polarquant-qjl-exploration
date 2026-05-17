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
    src = root / "csrc" / "turboquant_logits_tree_cuda.cu"

    _EXT = load(
        name="turboquant_polar_tree_logits_cuda_ext",
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


@torch.no_grad()
def turboquant_polar_tree_fused_logits_cuda(
    *,
    q: torch.Tensor,
    q_projected: torch.Tensor,

    packed_meta: torch.Tensor,
    radii: torch.Tensor,

    centroids_l1: torch.Tensor,
    centroids_l2: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,

    qjl_norms: torch.Tensor,
) -> torch.Tensor:
    """
    PolarQuant meta64 direct-tree logits fast path.

    This computes the Polar Stage-1 contribution as a recursive tree dot
    product instead of reconstructing each coordinate first. The compressed
    format is intentionally unchanged from the existing 5.125 physical-bpc
    PolarQuant benchmark:

      - D = 128
      - QJL M = 128
      - bits_by_level = (4,2,2,2)
      - B = 1, Q = 1
      - packed_meta = [B,H,T,64]

    Returns:
        [B,H,Q,T] float32 logits.
    """
    ext = _load_ext()

    if not q.is_cuda:
        raise ValueError("q must be CUDA.")
    if not q_projected.is_cuda:
        raise ValueError("q_projected must be CUDA.")
    if not packed_meta.is_cuda:
        raise ValueError("packed_meta must be CUDA.")

    q_f32 = q.contiguous().to(torch.float32)
    qproj_f32 = q_projected.contiguous().to(torch.float32)
    meta_u8 = packed_meta.contiguous()
    radii_f16 = radii.contiguous().to(torch.float16)
    qjl_norms_f16 = qjl_norms.contiguous().to(torch.float16)

    c1 = centroids_l1.contiguous().to(torch.float32)
    c2 = centroids_l2.contiguous().to(torch.float32)
    c3 = centroids_l3.contiguous().to(torch.float32)
    c4 = centroids_l4.contiguous().to(torch.float32)

    cos_l1 = torch.cos(c1).contiguous()
    sin_l1 = torch.sin(c1).contiguous()
    cos_l2 = torch.cos(c2).contiguous()
    sin_l2 = torch.sin(c2).contiguous()
    cos_l3 = torch.cos(c3).contiguous()
    sin_l3 = torch.sin(c3).contiguous()
    cos_l4 = torch.cos(c4).contiguous()
    sin_l4 = torch.sin(c4).contiguous()

    return ext.turboquant_polar_tree_fused_logits_cuda(
        q_f32,
        qproj_f32,
        meta_u8,
        radii_f16,
        cos_l1,
        sin_l1,
        cos_l2,
        sin_l2,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        qjl_norms_f16,
    )
