from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        src = (
            Path(__file__).resolve().parent
            / "csrc"
            / "turboquant_custom_candidate_selector_cuda.cu"
        )
        _EXT = load(
            name="turboquant_custom_candidate_selector_cuda_ext",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


@torch.no_grad()
def turboquant_custom_candidate_topk_selector_cuda(
    polar_logits: torch.Tensor,
    *,
    topk: int = 128,
) -> torch.Tensor:
    """
    Two-stage custom CUDA candidate selector.

    Input:
      polar_logits: [1,H,1,T] float32

    Output:
      selected_indices: [1,H,1,K] int64

    Structure:
      1) warp-group local top-8 from each 128-token segment
      2) exact per-head merge top-K over the local candidate pool

    The local stage is approximate by design; evaluate candidate recall against
    the full fused reference before treating it as a production selector.
    """
    ext = _load_ext()
    return ext.turboquant_custom_candidate_topk_selector_cuda(
        polar_logits.contiguous().to(torch.float32),
        int(topk),
    )
