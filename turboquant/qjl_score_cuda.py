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
    src = root / "csrc" / "qjl_score_cuda.cu"

    _EXT = load(
        name="turboquant_qjl_score_cuda_ext",
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
def qjl_packed_score_cuda(
    *,
    q_projected: torch.Tensor,
    packed_signs: torch.Tensor,
    norms: torch.Tensor,
) -> torch.Tensor:
    """
    Fused QJL residual score.

    Args:
        q_projected:
            [B,H,Q,128], float32

        packed_signs:
            [B,H,T,16], uint8

        norms:
            [B,H,T], float16

    Returns:
        scores:
            [B,H,Q,T], float32
    """
    ext = _load_ext()

    if not q_projected.is_cuda:
        raise ValueError("q_projected must be CUDA.")

    return ext.qjl_packed_score_cuda(
        q_projected.contiguous().to(torch.float32),
        packed_signs.contiguous(),
        norms.contiguous().to(torch.float16),
    )
