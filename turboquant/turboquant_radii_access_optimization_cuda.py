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
    src = root / "csrc" / "turboquant_radii_access_optimization_cuda.cu"

    _EXT = load(
        name="turboquant_radii_access_optimization_cuda_ext",
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


def _validate_cuda(**tensors: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA.")


@torch.no_grad()
def turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda(
    *,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l2_factor_lut_fp16: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
) -> torch.Tensor:
    """
    Polar-only L2-combined-FP16-LUT path with radii load hoisted before the
    Polar tree critical path. This tests whether radii latency matters more
    than radii byte volume.
    """
    _validate_cuda(
        packed_meta=packed_meta,
        radii=radii,
        l2_factor_lut_fp16=l2_factor_lut_fp16,
    )
    ext = _load_ext()
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)
    return ext.turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda(
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        l2_factor_lut_fp16.contiguous().to(torch.float16),
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
    )


@torch.no_grad()
def turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_polar_only_cuda(
    *,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l2_factor_lut_fp16: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
) -> torch.Tensor:
    """
    Polar-only L2-combined-FP16-LUT path with warp-grouped radii pair loads:
      - lanes 0..3 load four packed half2 radii words
      - every lane receives the needed pair via warp shuffle
      - only lane4==0 consumes the unpacked radius
    """
    _validate_cuda(
        packed_meta=packed_meta,
        radii=radii,
        l2_factor_lut_fp16=l2_factor_lut_fp16,
    )
    ext = _load_ext()
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)
    return ext.turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_polar_only_cuda(
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        l2_factor_lut_fp16.contiguous().to(torch.float16),
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
    )


@torch.no_grad()
def turboquant_polar_tree_l2_combined_lut_fp16_early_radii_lane_nibble_fused_logits_cuda(
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
    Current best full fused path plus radii load hoisting.
    """
    _validate_cuda(
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=radii,
        l2_factor_lut_fp16=l2_factor_lut_fp16,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        qjl_norms=qjl_norms,
    )
    ext = _load_ext()
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)
    return ext.turboquant_polar_tree_l2_combined_lut_fp16_early_radii_lane_nibble_fused_logits_cuda(
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


@torch.no_grad()
def turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_lane_nibble_fused_logits_cuda(
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
    Current best full fused path plus warp-grouped radii half2 loads.
    """
    _validate_cuda(
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=radii,
        l2_factor_lut_fp16=l2_factor_lut_fp16,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        qjl_norms=qjl_norms,
    )
    ext = _load_ext()
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)
    return ext.turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_lane_nibble_fused_logits_cuda(
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
