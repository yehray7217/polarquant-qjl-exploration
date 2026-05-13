from __future__ import annotations

import torch
import turboquant_cuda


@torch.no_grad()
def turboquant_decode_score_transposed_cuda(
    *,
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    packed_mse_indices_t: torch.Tensor,
    mse_norms: torch.Tensor,
    packed_qjl_sign_bits_t: torch.Tensor,
    qjl_residual_norms: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    return turboquant_cuda.turboquant_decode_score_transposed(
        q_rot.contiguous(),
        q_sketch.contiguous(),
        packed_mse_indices_t.contiguous(),
        mse_norms.contiguous(),
        packed_qjl_sign_bits_t.contiguous(),
        qjl_residual_norms.contiguous(),
        centroids.contiguous(),
    )

@torch.no_grad()
def turboquant_decode_score_transposed_sharedq_cuda(
    *,
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    packed_mse_indices_t: torch.Tensor,
    mse_norms: torch.Tensor,
    packed_qjl_sign_bits_t: torch.Tensor,
    qjl_residual_norms: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    return turboquant_cuda.turboquant_decode_score_transposed_sharedq(
        q_rot.contiguous(),
        q_sketch.contiguous(),
        packed_mse_indices_t.contiguous(),
        mse_norms.contiguous(),
        packed_qjl_sign_bits_t.contiguous(),
        qjl_residual_norms.contiguous(),
        centroids.contiguous(),
    )