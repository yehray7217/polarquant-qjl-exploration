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
    src = root / "csrc" / "polar_reconstruct_cuda.cu"

    _EXT = load(
        name="turboquant_polar_reconstruct_cuda_ext",
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
def polar_stage1_reconstruct_cuda(
    *,
    packed_l1: torch.Tensor,
    packed_l2: torch.Tensor,
    packed_l3: torch.Tensor,
    packed_l4: torch.Tensor,
    radii: torch.Tensor,
    centroids_l1: torch.Tensor,
    centroids_l2: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
) -> torch.Tensor:
    """
    Fused Polar Stage-1 reconstruction.

    Args:
        packed_l1:
            [B,H,T,32], uint8

        packed_l2:
            [B,H,T,8], uint8

        packed_l3:
            [B,H,T,4], uint8

        packed_l4:
            [B,H,T,2], uint8

        radii:
            [B,H,T,8], float16

    Returns:
        x_hat:
            [B,H,T,128], float32
    """
    ext = _load_ext()

    if not packed_l1.is_cuda:
        raise ValueError(
            "packed_l1 must be CUDA."
        )

    return ext.polar_stage1_reconstruct_cuda(
        packed_l1.contiguous(),
        packed_l2.contiguous(),
        packed_l3.contiguous(),
        packed_l4.contiguous(),
        radii.contiguous().to(torch.float16),
        centroids_l1.contiguous().to(torch.float32),
        centroids_l2.contiguous().to(torch.float32),
        centroids_l3.contiguous().to(torch.float32),
        centroids_l4.contiguous().to(torch.float32),
    )
