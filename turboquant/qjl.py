from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional

import math
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
class QJLEncoding:
    """
    QJL encoding for a batch of vectors.

    sign_bits:
        Sign sketch values in {-1,+1}, shape [N, M].

    norms:
        Original vector L2 norms, shape [N].
    """
    sign_bits: torch.Tensor
    norms: torch.Tensor


# ============================================================
# Gaussian sketch
# ============================================================

@torch.no_grad()
def make_gaussian_sketch(
    d: int,
    m: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Gaussian sketch matrix S with shape [M, D].

    The estimator below assumes rows distributed approximately as N(0, I).
    """
    device = torch.device(device)

    if seed is None:
        S = torch.randn(
            m,
            d,
            dtype=dtype,
            device=device,
        )
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        S = torch.randn(
            m,
            d,
            dtype=dtype,
            device=device,
            generator=gen,
        )

    return S.contiguous()


# ============================================================
# QJL encode
# ============================================================

@torch.no_grad()
def qjl_encode(
    x: torch.Tensor,
    S: torch.Tensor,
) -> QJLEncoding:
    """
    Encode x with 1-bit sign sketches.

    Input:
      x:
        [N, D]

      S:
        [M, D]

    Output:
      QJLEncoding(
          sign_bits=[N, M] in {-1,+1},
          norms=[N],
      )

    NVTX breakdown:
      - tq_qjl_project
      - tq_qjl_norm
      - tq_qjl_sign
      - tq_qjl_finalize
    """
    if x.ndim != 2:
        raise ValueError(
            f"x must be [N,D], got shape={tuple(x.shape)}"
        )

    if S.ndim != 2:
        raise ValueError(
            f"S must be [M,D], got shape={tuple(S.shape)}"
        )

    D = x.shape[-1]

    if S.shape[-1] != D:
        raise ValueError(
            f"sketch dimension mismatch: x D={D}, S={tuple(S.shape)}"
        )

    with _nvtx_range("tq_qjl_project"):
        x_fp = x.to(dtype=S.dtype)
        projected = x_fp @ S.T

    with _nvtx_range("tq_qjl_norm"):
        norms = torch.linalg.vector_norm(
            x_fp,
            ord=2,
            dim=-1,
        )

    with _nvtx_range("tq_qjl_sign"):
        sign_bits = torch.where(
            projected >= 0,
            torch.ones_like(projected),
            -torch.ones_like(projected),
        )

    with _nvtx_range("tq_qjl_finalize"):
        sign_bits = sign_bits.contiguous()
        norms = norms.contiguous()

    return QJLEncoding(
        sign_bits=sign_bits,
        norms=norms,
    )


# ============================================================
# QJL inner-product estimator
# ============================================================

@torch.no_grad()
def qjl_inner_product_estimate(
    q: torch.Tensor,
    encoded_r: QJLEncoding,
    S: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate <q, r> from:
      - query vector q
      - sign(S r)
      - ||r||

    Using the Gaussian identity:
      E[ sign(<s,r>) * <s,q> ]
        = sqrt(2/pi) * <q,r> / ||r||

    Hence:
      <q,r>
        ≈ sqrt(pi/2) * ||r|| *
          mean_j[ sign(<s_j,r>) * <s_j,q> ]

    Input:
      q:
        [N, D]

      encoded_r.sign_bits:
        [N, M]

      encoded_r.norms:
        [N]

      S:
        [M, D]

    Output:
      estimated dot products, shape [N]
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

    if D != D_s:
        raise ValueError(
            f"query/sketch dimension mismatch: q D={D}, S D={D_s}"
        )

    if encoded_r.sign_bits.shape != (N, M):
        raise ValueError(
            f"encoded_r.sign_bits must be {(N, M)}, "
            f"got {tuple(encoded_r.sign_bits.shape)}"
        )

    if encoded_r.norms.shape != (N,):
        raise ValueError(
            f"encoded_r.norms must be {(N,)}, "
            f"got {tuple(encoded_r.norms.shape)}"
        )

    q_fp = q.to(dtype=S.dtype)

    # [N, M]
    Sq = q_fp @ S.T

    # [N]
    signed_mean = torch.mean(
        Sq * encoded_r.sign_bits.to(dtype=Sq.dtype),
        dim=-1,
    )

    # [N]
    estimate = (
        math.sqrt(math.pi / 2.0)
        * encoded_r.norms.to(dtype=signed_mean.dtype)
        * signed_mean
    )

    return estimate