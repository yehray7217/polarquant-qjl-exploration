from __future__ import annotations

import torch
import turboquant_cuda


@torch.no_grad()
def turboquant_mse_lut_score_transposed_cuda(
    *,
    q_rot: torch.Tensor,
    packed_mse_indices_t: torch.Tensor,
    mse_norms: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    return turboquant_cuda.turboquant_mse_lut_score_transposed(
        q_rot.contiguous(),
        packed_mse_indices_t.contiguous(),
        mse_norms.contiguous(),
        centroids.contiguous(),
    )
