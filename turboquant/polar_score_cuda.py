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
    src = root / "csrc" / "polar_score_cuda.cu"

    _EXT = load(
        name="turboquant_polar_score_cuda_ext",
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
def polar_stage1_score_cuda(
    *,
    q: torch.Tensor,

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
    Fused Stage-1 Polar score.

    Input:
        q:
            [B,H,Q,128], float32

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

        centroids_l1:
            [16]

        centroids_l2:
            [4]

        centroids_l3:
            [4]

        centroids_l4:
            [4]

    Output:
        scores:
            [B,H,Q,T], float32

    Optimization:
        CUDA kernel no longer computes sin/cos internally.
        This wrapper precomputes:
            cos/sin of centroid tables
        and passes compact trig lookup tables to the kernel.
    """
    ext = _load_ext()

    if not q.is_cuda:
        raise ValueError(
            "q must be CUDA."
        )

    q_f32 = q.contiguous().to(torch.float32)

    packed_l1_u8 = packed_l1.contiguous()
    packed_l2_u8 = packed_l2.contiguous()
    packed_l3_u8 = packed_l3.contiguous()
    packed_l4_u8 = packed_l4.contiguous()

    radii_f16 = radii.contiguous().to(torch.float16)

    c1 = centroids_l1.contiguous().to(torch.float32)
    c2 = centroids_l2.contiguous().to(torch.float32)
    c3 = centroids_l3.contiguous().to(torch.float32)
    c4 = centroids_l4.contiguous().to(torch.float32)

    # ------------------------------------------------------------
    # Small trig lookup tables.
    #
    # Sizes:
    #   L1: 16
    #   L2: 4
    #   L3: 4
    #   L4: 4
    #
    # These are tiny; creating them here is far cheaper than
    # recomputing sinf/cosf inside every score kernel thread.
    # ------------------------------------------------------------

    cos_l1 = torch.cos(c1).contiguous()
    sin_l1 = torch.sin(c1).contiguous()

    cos_l2 = torch.cos(c2).contiguous()
    sin_l2 = torch.sin(c2).contiguous()

    cos_l3 = torch.cos(c3).contiguous()
    sin_l3 = torch.sin(c3).contiguous()

    cos_l4 = torch.cos(c4).contiguous()
    sin_l4 = torch.sin(c4).contiguous()

    return ext.polar_stage1_score_cuda(
        q_f32,

        packed_l1_u8,
        packed_l2_u8,
        packed_l3_u8,
        packed_l4_u8,

        radii_f16,

        cos_l1,
        sin_l1,

        cos_l2,
        sin_l2,

        cos_l3,
        sin_l3,

        cos_l4,
        sin_l4,
    )