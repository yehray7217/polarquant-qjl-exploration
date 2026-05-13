from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional

import torch


# ============================================================
# NVTX helper
# ============================================================

@contextmanager
def _nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


# ============================================================
# Encoding container
# ============================================================

@dataclass
class MSEEncoding:
    """
    2-bit TurboQuant_mse encoding.

    indices:
        Quantized centroid ids, shape [N, D], values in {0,1,2,3}.

    norms:
        Per-vector L2 norm, shape [N].
    """
    indices: torch.Tensor
    norms: torch.Tensor


# ============================================================
# Fixed TurboQuant 2-bit centroids
# ============================================================

def get_2bit_centroids(
    d: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    TurboQuant-style 2-bit centroids for normalized rotated vectors.

    Values correspond to:
      [-1.510, -0.453, 0.453, 1.510] / sqrt(d)

    For d=128:
      [-0.1334664, -0.0400399, 0.0400399, 0.1334664]
    """
    base = torch.tensor(
        [-1.510, -0.453, 0.453, 1.510],
        dtype=dtype,
        device=device,
    )

    return base / (float(d) ** 0.5)

def get_1bit_centroids(
    d: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    TurboQuant-style 1-bit centroids for normalized rotated vectors.

    We use the symmetric 1-bit Lloyd-Max codebook under the
    high-dimensional Gaussian-limit approximation:

      [-sqrt(2/pi), +sqrt(2/pi)] / sqrt(d)

    Numerically:
      [-0.79788456, 0.79788456] / sqrt(d)
    """
    base = torch.tensor(
        [-0.7978845608028654, 0.7978845608028654],
        dtype=dtype,
        device=device,
    )

    return base / (float(d) ** 0.5)


# ============================================================
# Random orthogonal rotation
# ============================================================

@torch.no_grad()
def make_random_rotation(
    d: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a deterministic random orthogonal rotation matrix [D, D].

    We sample a Gaussian matrix and take QR decomposition.
    """
    device = torch.device(device)

    if seed is None:
        A = torch.randn(
            d,
            d,
            dtype=dtype,
            device=device,
        )
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        A = torch.randn(
            d,
            d,
            dtype=dtype,
            device=device,
            generator=gen,
        )

    Q, R = torch.linalg.qr(A)

    # Standard sign correction so the sampled orthogonal basis is stable
    # under QR sign convention.
    diag = torch.diagonal(R)
    sign = torch.sign(diag)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    Q = Q * sign.unsqueeze(0)

    return Q.contiguous()


# ============================================================
# 2-bit MSE quantization
# ============================================================

@torch.no_grad()
def turboquant_mse_quantize_2bit(
    x: torch.Tensor,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
) -> MSEEncoding:
    """
    2-bit TurboQuant_mse encoder.

    Conceptual flow:
      1. Rotate x
      2. Compute per-vector L2 norm
      3. Normalize coordinates
      4. Assign each coordinate to nearest centroid
      5. Return centroid indices + original norm

    Input:
      x:
        [N, D]

      rotation:
        [D, D]

      centroids:
        [4]

    Output:
      MSEEncoding(
          indices=[N, D],
          norms=[N],
      )

    NVTX breakdown:
      - tq_mse_rotate
      - tq_mse_norm
      - tq_mse_normalize
      - tq_mse_centroid_assign
      - tq_mse_finalize
    """
    if x.ndim != 2:
        raise ValueError(
            f"x must be [N,D], got shape={tuple(x.shape)}"
        )

    if rotation.ndim != 2:
        raise ValueError(
            f"rotation must be [D,D], got shape={tuple(rotation.shape)}"
        )

    if centroids.ndim != 1 or centroids.numel() != 4:
        raise ValueError(
            f"centroids must be shape [4], got shape={tuple(centroids.shape)}"
        )

    D = x.shape[-1]

    if rotation.shape != (D, D):
        raise ValueError(
            f"rotation shape mismatch: expected {(D, D)}, "
            f"got {tuple(rotation.shape)}"
        )

    with _nvtx_range("tq_mse_rotate"):
        x_fp = x.to(dtype=rotation.dtype)
        x_rot = x_fp @ rotation.T

    with _nvtx_range("tq_mse_norm"):
        norms = torch.linalg.vector_norm(
            x_rot,
            ord=2,
            dim=-1,
        )

    with _nvtx_range("tq_mse_normalize"):
        safe_norms = torch.clamp(
            norms,
            min=torch.finfo(x_rot.dtype).eps,
        )
        x_norm = x_rot / safe_norms.unsqueeze(-1)

    with _nvtx_range("tq_mse_centroid_assign"):
        # [N, D, 1] - [1, 1, 4] -> [N, D, 4]
        distances = torch.abs(
            x_norm.unsqueeze(-1) - centroids.view(1, 1, 4)
        )

        indices = torch.argmin(
            distances,
            dim=-1,
        )

    with _nvtx_range("tq_mse_finalize"):
        indices = indices.contiguous()
        norms = norms.contiguous()

    return MSEEncoding(
        indices=indices,
        norms=norms,
    )


# ============================================================
# 2-bit MSE dequantization
# ============================================================

@torch.no_grad()
def turboquant_mse_dequantize_2bit(
    encoding: MSEEncoding,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct x_hat from 2-bit MSE encoding.

    Steps:
      centroid lookup
      scale by stored norm
      inverse-rotate back to original space

    Since quantization used:
      x_rot = x @ rotation.T

    reconstruction uses:
      x_hat = x_hat_rot @ rotation
    """
    indices = encoding.indices
    norms = encoding.norms

    if indices.ndim != 2:
        raise ValueError(
            f"encoding.indices must be [N,D], got shape={tuple(indices.shape)}"
        )

    if norms.ndim != 1:
        raise ValueError(
            f"encoding.norms must be [N], got shape={tuple(norms.shape)}"
        )

    N, D = indices.shape

    if norms.shape[0] != N:
        raise ValueError(
            f"norm count mismatch: N={N}, norms={tuple(norms.shape)}"
        )

    if rotation.shape != (D, D):
        raise ValueError(
            f"rotation shape mismatch: expected {(D, D)}, "
            f"got {tuple(rotation.shape)}"
        )

    # [N, D]
    x_hat_norm = centroids[indices]

    # [N, D]
    x_hat_rot = x_hat_norm * norms.to(
        dtype=x_hat_norm.dtype
    ).unsqueeze(-1)

    # [N, D]
    x_hat = x_hat_rot @ rotation

    return x_hat

def get_4bit_centroids(
    d: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    TurboQuant-style 4-bit scalar centroids for normalized rotated vectors.

    These are the symmetric 16-level Lloyd-Max centroids for a
    standard normal coordinate distribution, scaled by 1/sqrt(d),
    matching the same Gaussian-limit style used by the existing
    1-bit / 2-bit MSE codebooks.
    """
    base = torch.tensor(
        [
            -2.73258957,
            -2.06901723,
            -1.61804639,
            -1.25623120,
            -0.94234046,
            -0.65675912,
            -0.38804830,
            -0.12839503,
             0.12839503,
             0.38804830,
             0.65675912,
             0.94234046,
             1.25623120,
             1.61804639,
             2.06901723,
             2.73258957,
        ],
        dtype=dtype,
        device=device,
    )

    return base / (float(d) ** 0.5)