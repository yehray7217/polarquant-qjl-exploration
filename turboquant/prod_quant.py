from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager

import torch

from turboquant.mse_quant import (
    MSEEncoding,
    turboquant_mse_quantize_2bit,
    turboquant_mse_dequantize_2bit,
)
from turboquant.qjl import (
    QJLEncoding,
    qjl_encode,
    qjl_inner_product_estimate,
)
from turboquant.cuda_packing import (
    pack_sign_bits_cuda,
)
from turboquant.cuda_packing import (
    pack_sign_bits_cuda,
)
from turboquant.cuda_mse import (
    mse_assign_pack_reconstruct_rot_cuda,
)


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


@dataclass
class ProdEncoding:
    """
    TurboQuant_prod prototype encoding.

    mse:
        2-bit TurboQuant_mse encoding.

    qjl_residual:
        1-bit QJL encoding of residual:
          residual = x - dequant_mse(x)
    """
    mse: MSEEncoding
    qjl_residual: QJLEncoding

@dataclass
class RuntimePackedProdEncoding:
    """
    Runtime-only TurboQuant_prod encoding.

    Stores the final cache-ready compressed payload directly:
      - packed 2-bit MSE centroid indices
      - MSE norms
      - packed 1-bit QJL residual signs
      - QJL residual norms

    No intermediate int64 MSE indices or float32 QJL sign tensor
    is kept in the runtime hot path.
    """
    packed_mse_indices: torch.Tensor
    mse_norms: torch.Tensor
    packed_qjl_sign_bits: torch.Tensor
    qjl_residual_norms: torch.Tensor

@torch.no_grad()
def turboquant_prod_quantize_3bit(
    x: torch.Tensor,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
    sketch: torch.Tensor,
) -> ProdEncoding:
    """
    Prototype TurboQuant_prod with:
      - 2-bit TurboQuant_mse main reconstruction
      - 1-bit QJL residual correction

    The overall conceptual bit-width is 3 bits/channel,
    ignoring norm / metadata overhead for now.

    NVTX breakdown:
      - tq_quantize_mse_encode
      - tq_quantize_mse_reconstruct
      - tq_quantize_residual
      - tq_quantize_qjl_encode
    """

    with _nvtx_range("tq_quantize_mse_encode"):
        mse_enc = turboquant_mse_quantize_2bit(
            x=x,
            rotation=rotation,
            centroids=centroids,
        )

    with _nvtx_range("tq_quantize_mse_reconstruct"):
        x_hat_mse = turboquant_mse_dequantize_2bit(
            encoding=mse_enc,
            rotation=rotation,
            centroids=centroids,
        )

    with _nvtx_range("tq_quantize_residual"):
        residual = x.to(dtype=x_hat_mse.dtype) - x_hat_mse

    with _nvtx_range("tq_quantize_qjl_encode"):
        qjl_enc = qjl_encode(
            x=residual,
            S=sketch,
        )

    return ProdEncoding(
        mse=mse_enc,
        qjl_residual=qjl_enc,
    )

@torch.no_grad()
def turboquant_prod_quantize_3bit_runtime_packed_qjl(
    x: torch.Tensor,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
    sketch: torch.Tensor,
) -> RuntimePackedProdEncoding:
    """
    Runtime-optimized TurboQuant_prod encode path.

    Compared with the reference path:
      - MSE centroid assignment + 2-bit packing + x_hat_rot
        reconstruction are fused in CUDA.
      - QJL residual signs are packed directly from projected residuals,
        avoiding a materialized float32 {-1,+1} sign tensor.

    Final outputs are directly cache-ready:
      - packed_mse_indices
      - mse_norms
      - packed_qjl_sign_bits
      - qjl_residual_norms
    """

    # ============================================================
    # MSE encode:
    #   x -> x_rot -> norms -> x_norm
    #   -> fused CUDA assign + 2-bit pack + x_hat_rot
    # ============================================================

    with _nvtx_range("tq_quantize_mse_encode"):
        with _nvtx_range("tq_mse_rotate"):
            x_fp = x.to(dtype=rotation.dtype)
            x_rot = x_fp @ rotation.T

        with _nvtx_range("tq_mse_norm"):
            mse_norms = torch.linalg.vector_norm(
                x_rot,
                ord=2,
                dim=-1,
            )

        with _nvtx_range("tq_mse_normalize"):
            safe_norms = torch.clamp(
                mse_norms,
                min=torch.finfo(x_rot.dtype).eps,
            )
            x_norm = x_rot / safe_norms.unsqueeze(-1)

        with _nvtx_range("tq_mse_fused_assign_pack_xhatrot"):
            packed_mse_indices, x_hat_rot = (
                mse_assign_pack_reconstruct_rot_cuda(
                    x_norm=x_norm.contiguous(),
                    norms=mse_norms.contiguous(),
                    centroids=centroids.contiguous(),
                )
            )

    # ============================================================
    # MSE reconstruction:
    #   x_hat_rot -> inverse rotate -> x_hat_mse
    # ============================================================

    with _nvtx_range("tq_quantize_mse_reconstruct"):
        with _nvtx_range("tq_mse_inverse_rotate"):
            x_hat_mse = x_hat_rot @ rotation

    # ============================================================
    # Residual
    # ============================================================

    with _nvtx_range("tq_quantize_residual"):
        residual = x.to(dtype=x_hat_mse.dtype) - x_hat_mse

    # ============================================================
    # QJL runtime path:
    #   residual -> projection -> norm -> direct packed sign bits
    # ============================================================

    with _nvtx_range("tq_quantize_qjl_project"):
        projected = residual.to(dtype=sketch.dtype) @ sketch.T

    with _nvtx_range("tq_quantize_qjl_norm"):
        qjl_residual_norms = torch.linalg.vector_norm(
            residual.to(dtype=sketch.dtype),
            ord=2,
            dim=-1,
        )

    with _nvtx_range("tq_quantize_qjl_pack_direct"):
        packed_qjl_sign_bits = pack_sign_bits_cuda(
            projected.contiguous()
        )

    return RuntimePackedProdEncoding(
        packed_mse_indices=packed_mse_indices,
        mse_norms=mse_norms.contiguous(),
        packed_qjl_sign_bits=packed_qjl_sign_bits,
        qjl_residual_norms=qjl_residual_norms.contiguous(),
    )

@torch.no_grad()
def turboquant_prod_mse_reconstruction(
    encoding: ProdEncoding,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Return only the MSE reconstruction x_hat_mse.

    Useful for comparing MSE-only dot product against
    prod-corrected dot product.
    """
    return turboquant_mse_dequantize_2bit(
        encoding=encoding.mse,
        rotation=rotation,
        centroids=centroids,
    )


@torch.no_grad()
def turboquant_prod_inner_product_estimate(
    q: torch.Tensor,
    encoding: ProdEncoding,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
    sketch: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate <q, x> using:
      <q, x_hat_mse> + QJL(q, residual)
    """
    x_hat_mse = turboquant_prod_mse_reconstruction(
        encoding=encoding,
        rotation=rotation,
        centroids=centroids,
    )

    mse_dot = torch.sum(
        q.to(dtype=x_hat_mse.dtype) * x_hat_mse,
        dim=-1,
    )

    residual_correction = qjl_inner_product_estimate(
        q=q,
        encoded_r=encoding.qjl_residual,
        S=sketch,
    )

    return mse_dot + residual_correction