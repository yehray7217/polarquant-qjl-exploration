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
    src = root / "csrc" / "turboquant_predecoded_l2_exact_ablation_cuda.cu"

    _EXT = load(
        name="turboquant_predecoded_l2_exact_ablation_cuda_ext",
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
def turboquant_predecoded_l2_exact_fused_logits_cuda(
    *,
    q_projected: torch.Tensor,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l2_factor_lut_fp16: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    qjl_norms: torch.Tensor,
    predecoded_l2_codes: torch.Tensor,
) -> torch.Tensor:
    """
    Exact-value L2 decode ablation on top of the current best full fused path:
      - L2 combined factor LUT, FP16 storage
      - lane-nibble QJL signs
      - early radii load
      - early qjl_norm load
      - c2 loaded from exact predecoded uint8 L2 codes

    c1a/c1b remain on the original packed-meta L1 decode path, so the combined
    LUT index is identical to baseline.
    """
    _validate_cuda(
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=radii,
        l2_factor_lut_fp16=l2_factor_lut_fp16,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        qjl_norms=qjl_norms,
        predecoded_l2_codes=predecoded_l2_codes,
    )
    ext = _load_ext()
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)
    return ext.turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_predecoded_l2_exact_lane_nibble_fused_logits_cuda(
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
        predecoded_l2_codes.contiguous().to(torch.uint8),
    )
