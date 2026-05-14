from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from turboquant.qjl_packing import (
    pack_qjl_signs_1bit,
    unpack_qjl_signs_1bit,
)

from turboquant.nvtx_utils import nvtx_range

QJL_CORRECTION_SCALE = 0.375

@dataclass
class QJLEncoding:
    """
    Packed QJL residual representation.

    packed_sign_bits:
        [N, M/8] uint8
        Stores sign(S r_unit) using one bit per sketch coordinate.

    norms:
        [N] float32
        Original residual norms ||r||_2.
    """
    packed_sign_bits: torch.Tensor
    norms: torch.Tensor


@torch.no_grad()
def make_gaussian_sketch(
    *,
    d: int,
    m: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Build Gaussian JL sketch S with shape [M, D].

    We use:
        S_ij ~ N(0, 1/M)

    so:
        S = randn(M, D) / sqrt(M)
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}.")

    if m <= 0:
        raise ValueError(f"m must be positive, got {m}.")

    generator = None

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    S = torch.randn(
        m,
        d,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    S = S / math.sqrt(float(m))

    return S.contiguous()


@torch.no_grad()
def qjl_encode(
    *,
    x: torch.Tensor,
    S: torch.Tensor,
    max_chunk_rows: int = 262_144,
) -> QJLEncoding:
    """
    Encode residual vectors with QJL using chunked projection.

    Args:
        x:
            [N, D]

        S:
            [M, D]

        max_chunk_rows:
            Number of vectors processed per chunk.
            This prevents materializing a huge [N, M]
            projected tensor for long-context K caches.

    Returns:
        QJLEncoding(
            packed_sign_bits=[N, M/8],
            norms=[N],
        )
    """
    if x.ndim != 2:
        raise ValueError(
            f"x must be [N,D], got shape={tuple(x.shape)}"
        )

    if S.ndim != 2:
        raise ValueError(
            f"S must be [M,D], got shape={tuple(S.shape)}"
        )

    if max_chunk_rows <= 0:
        raise ValueError(
            f"max_chunk_rows must be positive, got {max_chunk_rows}."
        )

    N, D = x.shape
    M, D_s = S.shape

    if int(D) != int(D_s):
        raise ValueError(
            f"x.shape[-1]={D} does not match S.shape[-1]={D_s}."
        )

    if int(M) % 8 != 0:
        raise ValueError(
            f"QJL sketch dimension M must be divisible by 8, got M={M}."
        )

    x_f = (
        x
        if x.dtype == torch.float32
        else x.to(torch.float32)
    )

    S_f = S.to(
        device=x.device,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------
    # Norms are only [N], cheap enough to compute globally.
    # Store fp16 as before.
    # ------------------------------------------------------------
    norms_fp32 = torch.linalg.vector_norm(
        x_f,
        ord=2,
        dim=-1,
    )

    safe_norms = torch.clamp(
        norms_fp32,
        min=torch.finfo(torch.float32).eps,
    )

    norms_store = norms_fp32.to(
        torch.float16
    ).contiguous()

    # ------------------------------------------------------------
    # Allocate final packed 1-bit signs once:
    #   [N, M/8]
    # ------------------------------------------------------------
    packed_sign_bits = torch.empty(
        int(N),
        int(M // 8),
        device=x.device,
        dtype=torch.uint8,
    )

    # ------------------------------------------------------------
    # Chunked QJL projection:
    #
    # For each chunk:
    #   x_unit_chunk   [C,D]
    #   projected      [C,M]
    #   sign_bits      [C,M] bool
    #   packed_signs   [C,M/8] uint8
    # ------------------------------------------------------------
    for start in range(
        0,
        int(N),
        int(max_chunk_rows),
    ):
        end = min(
            start + int(max_chunk_rows),
            int(N),
        )

        x_chunk = x_f[start:end]
        norm_chunk = safe_norms[start:end]

        x_unit_chunk = x_chunk / norm_chunk.unsqueeze(-1)

        projected_chunk = x_unit_chunk @ S_f.T

        sign_bits_chunk = projected_chunk >= 0

        packed_chunk = pack_qjl_signs_1bit(
            sign_bits_chunk
        )

        packed_sign_bits[start:end].copy_(
            packed_chunk
        )

        # Release large per-chunk temporaries eagerly.
        del x_chunk
        del norm_chunk
        del x_unit_chunk
        del projected_chunk
        del sign_bits_chunk
        del packed_chunk

    return QJLEncoding(
        packed_sign_bits=packed_sign_bits.contiguous(),
        norms=norms_store,
    )

@torch.no_grad()
def qjl_inner_product_estimate(
    *,
    q: torch.Tensor,
    encoded_r: QJLEncoding,
    S: torch.Tensor,
) -> torch.Tensor:
    """
    Asymmetric QJL inner-product estimator.

    Estimate:
        <q, r>

    where only r is sign-quantized.

    Args:
        q:
            [N, D]

        encoded_r:
            packed QJL encoding of residual r.

        S:
            [M, D]

    Returns:
        [N]
    """
    if q.ndim != 2:
        raise ValueError(
            f"q must be [N,D], got shape={tuple(q.shape)}"
        )

    if S.ndim != 2:
        raise ValueError(
            f"S must be [M,D], got shape={tuple(S.shape)}"
        )

    N, D = q.shape
    M, D_s = S.shape

    if int(D) != int(D_s):
        raise ValueError(
            f"q.shape[-1]={D} does not match S.shape[-1]={D_s}."
        )

    if encoded_r.norms.ndim != 1:
        raise ValueError(
            f"encoded_r.norms must be [N], got shape={tuple(encoded_r.norms.shape)}"
        )

    if int(encoded_r.norms.shape[0]) != int(N):
        raise ValueError(
            f"encoded_r.norms N mismatch: "
            f"{encoded_r.norms.shape[0]} vs q N={N}"
        )

    q_f = q.to(torch.float32)
    S_f = S.to(torch.float32)

    # [N, M]
    with nvtx_range("tq_qjl_project_q"):
        q_projected = q_f @ S_f.T

    # packed bits -> bool bits -> numeric ±1 signs
    with nvtx_range("tq_qjl_unpack_signs"):
        unpacked_bits = unpack_qjl_signs_1bit(
            encoded_r.packed_sign_bits
        )

    with nvtx_range("tq_qjl_bits_to_signs"):
        residual_signs = torch.where(
            unpacked_bits,
            torch.ones_like(
                unpacked_bits,
                dtype=torch.float32,
            ),
            -torch.ones_like(
                unpacked_bits,
                dtype=torch.float32,
            ),
        )

    # Since S ~ N(0, 1/M), the unbiased asymmetric estimator is:
    #
    #   sqrt(pi/2) * ||r|| * sum_j [ (Sq)_j * sign((Sr)_j) ] / sqrt(M)
    #
    with nvtx_range("tq_qjl_reduce_estimate"):
        correction = (
            QJL_CORRECTION_SCALE
            * math.sqrt(math.pi / 2.0)
            * encoded_r.norms.to(torch.float32)
            * torch.sum(
                q_projected * residual_signs,
                dim=-1,
            )
            / math.sqrt(float(M))
        )

    return correction.contiguous()