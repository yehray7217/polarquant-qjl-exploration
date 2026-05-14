from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from turboquant.nvtx_utils import nvtx_range

@dataclass
class PolarEncoding:
    """
    Recursive polar representation.

    angles:
        List of angle tensors.

        Level 1:
            shape [..., d/2]
            range [0, 2pi)

        Level l >= 2:
            shape [..., d / 2^l]
            range [0, pi/2]

    radii:
        Remaining radii after the requested number of recursive levels.

        If num_levels = log2(d):
            radii.shape = [...]

        If num_levels < log2(d):
            radii.shape = [..., d / 2^num_levels]

        Example:
            d = 128, num_levels = 4
            radii.shape = [..., 8]
    """
    angles: List[torch.Tensor]
    radii: torch.Tensor
    original_dim: int
    num_levels: int


def _check_power_of_two(d: int) -> None:
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(
            f"PolarQuant requires d to be a positive power of two, got d={d}."
        )


def _max_levels_for_dim(d: int) -> int:
    _check_power_of_two(d)
    return d.bit_length() - 1


@torch.no_grad()
def recursive_polar_encode(
    x: torch.Tensor,
    *,
    num_levels: int | None = None,
) -> PolarEncoding:
    """
    Recursive Cartesian -> polar transform.

    Args:
        x:
            Tensor of shape [..., d], where d is a power of two.

        num_levels:
            Number of recursive polar levels.

            - None:
                recurse all the way until one final radius remains.
            - integer L:
                stop after L polar levels.

    Returns:
        PolarEncoding(
            angles=[psi_1, psi_2, ..., psi_L],
            radii=remaining_radii,
            original_dim=d,
            num_levels=L,
        )

    Level 1:
        Pair Cartesian coordinates:
            (x0, x1), (x2, x3), ...

        Angle:
            atan2(y, x) mapped into [0, 2pi)

        Radius:
            sqrt(x^2 + y^2)

    Higher levels:
        Pair nonnegative radii:
            (r0, r1), (r2, r3), ...

        Angle:
            atan2(r_right, r_left) in [0, pi/2]

        Radius:
            sqrt(r_left^2 + r_right^2)
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension.")

    d = int(x.shape[-1])
    max_levels = _max_levels_for_dim(d)

    if num_levels is None:
        num_levels = max_levels

    if not isinstance(num_levels, int):
        raise TypeError("num_levels must be an int or None.")

    if num_levels < 1 or num_levels > max_levels:
        raise ValueError(
            f"num_levels must be in [1, {max_levels}], got {num_levels}."
        )

    angles: list[torch.Tensor] = []

    # ============================================================
    # Level 1: Cartesian coordinates -> polar
    # ============================================================

    left = x[..., 0::2]
    right = x[..., 1::2]

    psi_level_1 = torch.atan2(right, left)

    # atan2 returns [-pi, pi], convert to [0, 2pi)
    psi_level_1 = torch.remainder(
        psi_level_1,
        2.0 * torch.pi,
    )

    radii = torch.sqrt(
        torch.clamp(
            left * left + right * right,
            min=0.0,
        )
    )

    angles.append(psi_level_1)

    # ============================================================
    # Levels 2..L: recursive polar transform on radii
    # ============================================================

    for _level in range(2, num_levels + 1):
        r_left = radii[..., 0::2]
        r_right = radii[..., 1::2]

        psi = torch.atan2(
            r_right,
            r_left,
        )

        parent_radii = torch.sqrt(
            torch.clamp(
                r_left * r_left + r_right * r_right,
                min=0.0,
            )
        )

        angles.append(psi)
        radii = parent_radii

    # If fully recursive, squeeze final trailing radius dimension.
    # Otherwise preserve remaining radii axis.
    if num_levels == max_levels:
        radii_out = radii.squeeze(-1)
    else:
        radii_out = radii

    return PolarEncoding(
        angles=angles,
        radii=radii_out,
        original_dim=d,
        num_levels=num_levels,
    )


@torch.no_grad()
def recursive_polar_decode(
    encoding: PolarEncoding,
) -> torch.Tensor:
    """
    Recursive polar -> Cartesian inverse transform.

    Supports both:
        - fully recursive encoding
        - partially recursive encoding, e.g. num_levels=4

    Returns:
        x_hat with shape [..., original_dim]
    """
    angles = encoding.angles
    num_levels = int(encoding.num_levels)
    original_dim = int(encoding.original_dim)

    max_levels = _max_levels_for_dim(original_dim)

    if len(angles) != num_levels:
        raise ValueError(
            f"Expected {num_levels} angle tensors, got {len(angles)}."
        )

    if num_levels < 1 or num_levels > max_levels:
        raise ValueError(
            f"Invalid num_levels={num_levels} for original_dim={original_dim}."
        )

    # ============================================================
    # Restore top-level radii shape
    # ============================================================

    with nvtx_range("tq_polar_decode_tree"):
        if num_levels == max_levels:
            radii = encoding.radii.unsqueeze(-1)
        else:
            radii = encoding.radii

        for psi in reversed(angles[1:]):
            r_left = radii * torch.cos(psi)
            r_right = radii * torch.sin(psi)

            radii = torch.stack(
                [r_left, r_right],
                dim=-1,
            ).flatten(start_dim=-2)

        psi_level_1 = angles[0]

        x_left = radii * torch.cos(psi_level_1)
        x_right = radii * torch.sin(psi_level_1)

        x = torch.stack(
            [x_left, x_right],
            dim=-1,
        ).flatten(start_dim=-2)

        if x.shape[-1] != original_dim:
            raise RuntimeError(
                f"Decoded dimension mismatch: expected {original_dim}, got {x.shape[-1]}."
            )

        return x