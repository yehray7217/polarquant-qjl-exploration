from __future__ import annotations

import torch


def _validate_query_shape(q: torch.Tensor) -> tuple[int, int]:
    if q.ndim != 4:
        raise ValueError(f"q must be [B,H,Q,D], got {tuple(q.shape)}.")
    B, H, Q, D = [int(x) for x in q.shape]
    if B != 1 or Q != 1 or D != 128:
        raise ValueError(
            "Current Polar tree LUT builders require "
            f"B=1, Q=1, D=128, got B={B}, Q={Q}, D={D}."
        )
    return H, D


def _trig_tables(centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = centroids.contiguous().to(torch.float32)
    return torch.cos(c).contiguous(), torch.sin(c).contiguous()


@torch.no_grad()
def build_tree_l1_factor_lut(
    *,
    q: torch.Tensor,
    centroids_l1: torch.Tensor,
) -> torch.Tensor:
    """
    Build query-side L1 factor LUT for PolarQuant direct-tree scoring.

    Output:
        [H, 64, 16] float32

    For each head h, L1 pair p, and L1 code c:
        LUT[h,p,c] =
            q[h, 2p]   * cos(theta_c)
          + q[h, 2p+1] * sin(theta_c)

    This table is query-dependent and key-independent, so it can be reused
    across all compressed keys at one decode step.
    """
    H, _ = _validate_query_shape(q)

    q_pairs = (
        q.contiguous()
        .to(torch.float32)[0, :, 0, :]
        .reshape(H, 64, 2)
    )

    cos_l1, sin_l1 = _trig_tables(centroids_l1)
    if int(cos_l1.numel()) != 16:
        raise ValueError(f"centroids_l1 must have 16 entries, got {cos_l1.numel()}.")

    lut = (
        q_pairs[..., 0].unsqueeze(-1) * cos_l1.view(1, 1, 16)
        + q_pairs[..., 1].unsqueeze(-1) * sin_l1.view(1, 1, 16)
    )

    return lut.contiguous()


@torch.no_grad()
def build_tree_l2_factor_lut(
    *,
    q: torch.Tensor,
    centroids_l1: torch.Tensor,
    centroids_l2: torch.Tensor,
) -> torch.Tensor:
    """
    Build query-side L2 factor LUT for PolarQuant direct-tree scoring.

    Output:
        [H, 32, 1024] float32

    Each row corresponds to one 4-coordinate group. The flattened code index is:
        combo = (c2 << 8) | (c1b << 4) | c1a

    where:
        c1a in [0, 15]
        c1b in [0, 15]
        c2  in [0,  3]

    The lookup value equals the tree's s2 value for that code combination:
        s2 = s1a * cos(theta2[c2]) + s1b * sin(theta2[c2])
    """
    H, _ = _validate_query_shape(q)

    q_groups = (
        q.contiguous()
        .to(torch.float32)[0, :, 0, :]
        .reshape(H, 32, 4)
    )

    cos_l1, sin_l1 = _trig_tables(centroids_l1)
    cos_l2, sin_l2 = _trig_tables(centroids_l2)

    if int(cos_l1.numel()) != 16:
        raise ValueError(f"centroids_l1 must have 16 entries, got {cos_l1.numel()}.")
    if int(cos_l2.numel()) != 4:
        raise ValueError(f"centroids_l2 must have 4 entries, got {cos_l2.numel()}.")

    s1a = (
        q_groups[..., 0].unsqueeze(-1) * cos_l1.view(1, 1, 16)
        + q_groups[..., 1].unsqueeze(-1) * sin_l1.view(1, 1, 16)
    )
    s1b = (
        q_groups[..., 2].unsqueeze(-1) * cos_l1.view(1, 1, 16)
        + q_groups[..., 3].unsqueeze(-1) * sin_l1.view(1, 1, 16)
    )

    # Shape convention before flatten:
    #   [H, 32, c2, c1b, c1a]
    # The contiguous reshape therefore matches:
    #   combo = (c2 << 8) | (c1b << 4) | c1a
    lut = (
        s1a[:, :, None, None, :] * cos_l2.view(1, 1, 4, 1, 1)
        + s1b[:, :, None, :, None] * sin_l2.view(1, 1, 4, 1, 1)
    )

    return lut.contiguous().reshape(H, 32, 1024)
