from __future__ import annotations

import torch

import turboquant_cuda


@torch.no_grad()
def turboquant_decode_score_cuda_from_cache(
    query_states: torch.Tensor,   # [B,H,1,D]
    cache,
    layer_idx: int,
) -> torch.Tensor:
    """
    Correctness-first CUDA decode score path.

    query_states:
      [B,H,1,D]

    returns:
      [B,H,1,T]
    """
    if query_states.ndim != 4:
        raise ValueError(
            f"query_states must be [B,H,1,D], got {tuple(query_states.shape)}"
        )

    if query_states.shape[2] != 1:
        raise ValueError(
            "This first CUDA kernel is decode-only and expects query length = 1."
        )

    layer = cache.layers[layer_idx]

    if layer.is_empty():
        raise RuntimeError(f"Layer {layer_idx} cache is empty.")

    q = query_states[:, :, 0, :].float().contiguous()  # [B,H,D]

    combined = torch.matmul(
        q,
        cache.combined_query_transform.T,
    ).contiguous()

    D = cache.rotation.shape[0]

    q_rot = combined[..., :D].contiguous()
    sq = combined[..., D:].contiguous()

    scores = turboquant_cuda.turboquant_decode_score(
        q_rot,
        sq,
        layer.packed_mse_indices_buffer.contiguous(),
        layer.mse_norms_buffer.contiguous(),
        layer.packed_qjl_sign_bits_buffer.contiguous(),
        layer.qjl_residual_norms_buffer.contiguous(),
        cache.centroids.contiguous(),
        int(layer.seq_len()),
    )

    return scores.unsqueeze(2)  # [B,H,1,T]
