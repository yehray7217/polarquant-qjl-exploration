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
    src = root / "csrc" / "turboquant_logits_tree_predication_cuda.cu"

    _EXT = load(
        name="turboquant_polar_tree_predication_cuda_ext",
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


def _prepare_common_inputs(
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
):
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

    return (
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


@torch.no_grad()
def turboquant_polar_tree_remap_s4_fused_logits_cuda(
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
    Predication-reduction experiment A.

    Keeps the 5.125 physical-bpc meta64 PolarQuant format and eliminates
    the final tree-level lane selection by duplicating the level-4 combine
    across the four lanes assigned to each 16-coordinate Polar subtree.
    The four duplicated contributions are scaled by 1/4 before warp
    reduction, preserving exact logits up to normal floating-point order.
    """
    ext = _load_ext()
    args = _prepare_common_inputs(
        q=q,
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=radii,
        centroids_l1=centroids_l1,
        centroids_l2=centroids_l2,
        centroids_l3=centroids_l3,
        centroids_l4=centroids_l4,
        qjl_norms=qjl_norms,
    )
    return ext.turboquant_polar_tree_remap_s4_fused_logits_cuda(*args)


@torch.no_grad()
def turboquant_polar_tree_remap_s34_fused_logits_cuda(
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
    Predication-reduction experiment B.

    Keeps the 5.125 physical-bpc meta64 PolarQuant format and duplicates
    both the level-3 and level-4 combines so all four lanes in each
    16-coordinate subtree execute the late-tree arithmetic. The duplicated
    final contributions are scaled by 1/4 before warp reduction.
    """
    ext = _load_ext()
    args = _prepare_common_inputs(
        q=q,
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=radii,
        centroids_l1=centroids_l1,
        centroids_l2=centroids_l2,
        centroids_l3=centroids_l3,
        centroids_l4=centroids_l4,
        qjl_norms=qjl_norms,
    )
    return ext.turboquant_polar_tree_remap_s34_fused_logits_cuda(*args)
