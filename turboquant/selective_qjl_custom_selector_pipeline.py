from __future__ import annotations

from dataclasses import dataclass

import torch

from turboquant.turboquant_radii_access_optimization_cuda import (
    turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda,
)
from turboquant.turboquant_selected_qjl_refine_cuda import (
    turboquant_selected_qjl_refine_topk_m128_cuda,
)
from turboquant.turboquant_custom_candidate_selector_cuda import (
    turboquant_custom_candidate_topk_selector_cuda,
)


@dataclass(frozen=True)
class SelectiveQJLResult:
    polar_logits: torch.Tensor
    selected_indices: torch.Tensor
    selected_refined_logits: torch.Tensor


def _validate_topk(topk: int, seq_len: int) -> int:
    k = int(topk)
    if k <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")
    if k > 128:
        raise ValueError("custom selector mainline currently supports topk <= 128.")
    return min(k, int(seq_len))


@torch.no_grad()
def selective_qjl_custom_selector_sparse_topk_m128_cuda(
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
    New candidate-selector mainline:
      Polar-only retrieval
      -> custom CUDA approximate top-K selector
      -> selected QJL refinement
    """
    polar_logits = turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda(
        packed_meta=packed_meta,
        radii=radii,
        l2_factor_lut_fp16=l2_factor_lut_fp16,
        centroids_l3=centroids_l3,
        centroids_l4=centroids_l4,
    )
    if polar_logits.ndim != 4:
        raise RuntimeError(
            f"Expected polar_logits [1,H,1,T], got {tuple(polar_logits.shape)}."
        )

    k = _validate_topk(topk, polar_logits.shape[-1])
    selected_indices = turboquant_custom_candidate_topk_selector_cuda(
        polar_logits,
        topk=k,
    )
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
def selective_qjl_custom_selector_dense_logits_topk_m128_cuda(
    **kwargs,
) -> torch.Tensor:
    result = selective_qjl_custom_selector_sparse_topk_m128_cuda(**kwargs)
    dense = result.polar_logits.clone()
    dense.scatter_(
        dim=-1,
        index=result.selected_indices,
        src=result.selected_refined_logits,
    )
    return dense
