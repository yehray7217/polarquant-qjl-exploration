from __future__ import annotations

from dataclasses import dataclass

import torch

from turboquant.turboquant_radii_access_optimization_cuda import (
    turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda,
)
from turboquant.turboquant_selected_qjl_refine_cuda import (
    turboquant_selected_qjl_refine_topk_m128_cuda,
)


@dataclass(frozen=True)
class SelectiveQJLResult:
    """
    Result container for the new selective-QJL inference mainline.

    `polar_logits` is the dense Polar-only retrieval score over all tokens; current CUDA paths typically return [B,H,Q,T].
    `selected_indices` are the top-K candidate token indices selected from
    `polar_logits`.
    `selected_refined_logits` are the Polar+QJL refined logits only for those
    selected candidates.
    """
    polar_logits: torch.Tensor
    selected_indices: torch.Tensor
    selected_refined_logits: torch.Tensor


def _validate_topk(topk: int, seq_len: int) -> int:
    k = int(topk)
    if k <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")
    return min(k, int(seq_len))


@torch.no_grad()
def selective_qjl_sparse_topk_m128_cuda(
    *,
    q_projected: torch.Tensor,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l2_factor_lut_fp16: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    qjl_norms: torch.Tensor,
    topk: int = 128,
) -> SelectiveQJLResult:
    """
    New selective-QJL mainline:

      1) Polar-only dense retrieval over all T keys
      2) top-K candidate selection using torch.topk CUDA
      3) selected QJL refinement using the custom CUDA kernel

    The returned representation is sparse in the refined stage:
      - dense Polar logits are available for all tokens
      - refined Polar+QJL logits are only computed for selected candidates

    This is the preferred path for candidate-retrieval / rerank-style
    attention experiments.
    """
    polar_logits = turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda(
        packed_meta=packed_meta,
        radii=radii,
        l2_factor_lut_fp16=l2_factor_lut_fp16,
        centroids_l3=centroids_l3,
        centroids_l4=centroids_l4,
    )

    if polar_logits.ndim not in (3, 4):
        raise RuntimeError(
            "Expected polar_logits [B,H,T] or [B,H,Q,T], "
            f"got shape {tuple(polar_logits.shape)}."
        )

    # The current CUDA Polar-only path returns [B,H,Q,T]; all candidate
    # selection and selected-refinement logic operates along the last token
    # dimension, so both [B,H,T] and [B,H,Q,T] are valid.
    k = _validate_topk(topk, polar_logits.shape[-1])
    selected_indices = torch.topk(
        polar_logits,
        k=k,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices.contiguous()

    selected_refined_logits = turboquant_selected_qjl_refine_topk_m128_cuda(
        q_projected=q_projected,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        qjl_norms=qjl_norms,
        polar_logits=polar_logits,
        selected_indices=selected_indices,
    )

    return SelectiveQJLResult(
        polar_logits=polar_logits,
        selected_indices=selected_indices,
        selected_refined_logits=selected_refined_logits,
    )


@torch.no_grad()
def selective_qjl_dense_logits_topk_m128_cuda(
    *,
    q_projected: torch.Tensor,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l2_factor_lut_fp16: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    qjl_norms: torch.Tensor,
    topk: int = 128,
) -> torch.Tensor:
    """
    Convenience wrapper that materializes a dense approximate score tensor:

      dense_out = Polar-only logits everywhere
      dense_out[selected top-K] = selected Polar+QJL refined logits

    Use this only when an existing evaluation path expects dense logits.
    For production-style retrieval/rerank, prefer
    `selective_qjl_sparse_topk_m128_cuda`.
    """
    result = selective_qjl_sparse_topk_m128_cuda(
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=radii,
        l2_factor_lut_fp16=l2_factor_lut_fp16,
        centroids_l3=centroids_l3,
        centroids_l4=centroids_l4,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        qjl_norms=qjl_norms,
        topk=topk,
    )
    dense = result.polar_logits.clone()
    dense.scatter_(
        dim=-1,
        index=result.selected_indices,
        src=result.selected_refined_logits,
    )
    return dense
