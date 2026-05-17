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
    src = root / "csrc" / "turboquant_meta_segmented_decode_cuda.cu"

    _EXT = load(
        name="turboquant_meta_segmented_decode_cuda_ext",
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
def turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_trimmed_meta12_lane_nibble_fused_logits_cuda(
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
    Baseline math with packed-meta load trimmed from lane<16 to lane<12.
    Polar decode in this path consumes only packed-meta words 0..11.
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
    return ext.turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_trimmed_meta12_lane_nibble_fused_logits_cuda(
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
def turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_segmented_meta_l1l2_lane_nibble_fused_logits_cuda(
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
    Segmented packed-meta decode:
      - early L1/L2 meta words 0..9 before scalar early loads
      - late L3/L4 meta words 10..11 immediately before the Polar late tree
      - no unused packed-meta words 12..15
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
    return ext.turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_segmented_meta_l1l2_lane_nibble_fused_logits_cuda(
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
