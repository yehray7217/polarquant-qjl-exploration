from __future__ import annotations

from typing import Tuple

import torch

import turboquant_cuda


@torch.no_grad()
def mse_assign_pack_reconstruct_rot_cuda(
    x_norm: torch.Tensor,
    norms: torch.Tensor,
    centroids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused runtime MSE helper.

    Inputs:
      x_norm:    [N,D] float32 CUDA contiguous
      norms:     [N] float32 CUDA contiguous
      centroids: [4] float32 CUDA contiguous

    Returns:
      packed_mse_indices: [N,D/4] uint8
      x_hat_rot:          [N,D] float32
    """
    if not x_norm.is_cuda:
        raise ValueError("x_norm must be CUDA tensor")

    if not norms.is_cuda:
        raise ValueError("norms must be CUDA tensor")

    if not centroids.is_cuda:
        raise ValueError("centroids must be CUDA tensor")

    if x_norm.dtype != torch.float32:
        raise ValueError(
            f"x_norm must be float32, got {x_norm.dtype}"
        )

    if norms.dtype != torch.float32:
        raise ValueError(
            f"norms must be float32, got {norms.dtype}"
        )

    if centroids.dtype != torch.float32:
        raise ValueError(
            f"centroids must be float32, got {centroids.dtype}"
        )

    if not x_norm.is_contiguous():
        x_norm = x_norm.contiguous()

    if not norms.is_contiguous():
        norms = norms.contiguous()

    if not centroids.is_contiguous():
        centroids = centroids.contiguous()

    packed_mse_indices, x_hat_rot = (
        turboquant_cuda.mse_assign_pack_reconstruct_rot(
            x_norm,
            norms,
            centroids,
        )
    )

    return packed_mse_indices, x_hat_rot
