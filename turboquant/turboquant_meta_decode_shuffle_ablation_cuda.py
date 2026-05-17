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
    src = root / "csrc" / "turboquant_meta_decode_shuffle_ablation_cuda.cu"

    _EXT = load(
        name="turboquant_meta_decode_shuffle_ablation_cuda_ext",
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


def _common_call(
    ext_fn,
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
    _validate_cuda(
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=radii,
        l2_factor_lut_fp16=l2_factor_lut_fp16,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        qjl_norms=qjl_norms,
    )
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)
    return ext_fn(
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
def turboquant_probe_no_l1_meta_decode_cuda(**kwargs) -> torch.Tensor:
    """
    Latency/NCU probe only. Output intentionally differs.
    Removes the L1 packed-meta shuffle/decode chain from the current best
    full fused path while preserving the downstream structure.
    """
    ext = _load_ext()
    return _common_call(
        ext.turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_probe_no_l1_meta_decode_lane_nibble_fused_logits_cuda,
        **kwargs,
    )


@torch.no_grad()
def turboquant_probe_no_l2_meta_decode_cuda(**kwargs) -> torch.Tensor:
    """
    Latency/NCU probe only. Output intentionally differs.
    Removes the L2 packed-meta shuffle/decode chain from the current best
    full fused path while preserving the downstream structure.
    """
    ext = _load_ext()
    return _common_call(
        ext.turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_probe_no_l2_meta_decode_lane_nibble_fused_logits_cuda,
        **kwargs,
    )


@torch.no_grad()
def turboquant_probe_no_l1_l2_meta_decode_cuda(**kwargs) -> torch.Tensor:
    """
    Latency/NCU probe only. Output intentionally differs.
    Removes both L1 and L2 packed-meta shuffle/decode chains from the current
    best full fused path while preserving the downstream structure.
    """
    ext = _load_ext()
    return _common_call(
        ext.turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_probe_no_l1_l2_meta_decode_lane_nibble_fused_logits_cuda,
        **kwargs,
    )
