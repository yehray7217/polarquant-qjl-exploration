from __future__ import annotations

from dataclasses import dataclass

import torch

from turboquant.polarquant import (
    recursive_polar_encode,
    recursive_polar_decode,
)
from turboquant.polarquant_quant import (
    PolarAngleCodebooks,
    QuantizedPolarEncoding,
    quantize_polar_encoding,
    dequantize_polar_encoding,
)
from turboquant.qjl import (
    QJLEncoding,
    qjl_encode,
    qjl_inner_product_estimate,
)

from turboquant.nvtx_utils import nvtx_range

from turboquant.polar_score_cuda import (
    polar_stage1_score_cuda,
)

from turboquant.qjl_score_cuda import (
    qjl_packed_score_cuda,
)

from turboquant.polar_reconstruct_cuda import (
    polar_stage1_reconstruct_cuda,
)

from turboquant.turboquant_logits_cuda import (
    turboquant_fused_logits_cuda,
)

@dataclass
class PolarProdEncoding:
    """
    TurboQuant-style product encoding with:

      Stage 1:
        PolarQuant-style angle-quantized reconstruction.

      Stage 2:
        QJL encoding of residual:
          residual = x - x_hat_polar

    Note:
        qjl_residual is stored in flattened [N,D]-style QJL form,
        where:
          N = product of all prefix dimensions before D.

        Example:
          x: [B,H,T,D]
          residual_flat: [B*H*T, D]
    """
    polar: QuantizedPolarEncoding
    qjl_residual: QJLEncoding


def _flatten_last_dim(
    x: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...], int]:
    """
    Flatten arbitrary prefix dimensions into N.

    Input:
        x: [..., D]

    Returns:
        x_flat: [N, D]
        prefix_shape: original x.shape[:-1]
        D: last dimension
    """
    if x.ndim < 2:
        raise ValueError(
            f"Expected tensor with shape [..., D], got shape={tuple(x.shape)}"
        )

    prefix_shape = tuple(int(s) for s in x.shape[:-1])
    D = int(x.shape[-1])

    x_flat = x.reshape(
        -1,
        D,
    )

    return x_flat, prefix_shape, D


@torch.no_grad()
def turboquant_polar_prod_quantize(
    *,
    x: torch.Tensor,
    codebooks: PolarAngleCodebooks,
    sketch: torch.Tensor,
    num_levels: int = 4,
) -> PolarProdEncoding:
    """
    Encode x using:

      1. L-level recursive polar transform
      2. Per-level angle quantization
      3. Polar reconstruction
      4. QJL residual encoding

    Supports:
        x: [N,D]
        x: [B,H,T,D]
        or any [...,D] tensor.

    Internally, QJL residual encoding is performed on flattened:
        residual_flat: [N_total, D]
    """

    x_work = x.to(torch.float32)

    # ============================================================
    # Stage 1: Polar angle quantization
    # ============================================================

    polar_exact = recursive_polar_encode(
        x_work,
        num_levels=num_levels,
    )

    polar_quant = quantize_polar_encoding(
        encoding=polar_exact,
        codebooks=codebooks,
    )

    packed = polar_quant.packed_angles

    if packed.level1_4bit.ndim == 4:
        with nvtx_range("tq_polar_reconstruct_cuda_for_residual"):
            x_hat_polar = polar_stage1_reconstruct_cuda(
                packed_l1=packed.level1_4bit,
                packed_l2=packed.level2_2bit,
                packed_l3=packed.level3_2bit,
                packed_l4=packed.level4_2bit,
                radii=polar_quant.radii,
                centroids_l1=codebooks.centroids[0],
                centroids_l2=codebooks.centroids[1],
                centroids_l3=codebooks.centroids[2],
                centroids_l4=codebooks.centroids[3],
            )
    else:
        with nvtx_range("tq_polar_reconstruct_reference_for_residual"):
            polar_dequant = dequantize_polar_encoding(
                qencoding=polar_quant,
                codebooks=codebooks,
            )

            x_hat_polar = recursive_polar_decode(
                polar_dequant,
            )

    # ============================================================
    # Stage 2: QJL residual
    # ============================================================

    residual = x_work - x_hat_polar.to(torch.float32)

    residual_flat, _, _ = _flatten_last_dim(
        residual,
    )

    qjl_residual = qjl_encode(
        x=residual_flat,
        S=sketch,
    )

    return PolarProdEncoding(
        polar=polar_quant,
        qjl_residual=qjl_residual,
    )


@torch.no_grad()
def turboquant_polar_prod_reconstruction(
    *,
    encoding: PolarProdEncoding,
    codebooks: PolarAngleCodebooks,
) -> torch.Tensor:
    with nvtx_range("tq_polar_reconstruct_total"):
        polar_dequant = dequantize_polar_encoding(
            qencoding=encoding.polar,
            codebooks=codebooks,
        )

        return recursive_polar_decode(
            polar_dequant,
        )


@torch.no_grad()
def turboquant_polar_prod_inner_product_estimate(
    *,
    q: torch.Tensor,
    encoding: PolarProdEncoding,
    codebooks: PolarAngleCodebooks,
    sketch: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate <q, x> using:

      <q, x_hat_polar> + QJL(q, residual)

    Supports:
        q: [N,D]
        q: [B,H,T,D]
        or any [...,D] tensor.

    Returns:
        scores with shape q.shape[:-1]

    Examples:
        q: [N,D]       -> [N]
        q: [B,H,T,D]   -> [B,H,T]
    """

    q_work = q.to(torch.float32)

    # ============================================================
    # Stage-1 polar reconstruction dot product
    # ============================================================

    # ============================================================
    # Path A: 4D runtime / cache path
    #   q: [B,H,T,D]
    #
    # Uses fused Polar Stage-1 CUDA score, then extracts diagonal.
    # This is retained for direct chunkwise parity tests.
    # ============================================================

    if q_work.ndim == 4:
        packed = encoding.polar.packed_angles

        with nvtx_range("tq_polar_stage1_score_cuda"):
            polar_dot_bhqt = polar_stage1_score_cuda(
                q=q_work,
                packed_l1=packed.level1_4bit,
                packed_l2=packed.level2_2bit,
                packed_l3=packed.level3_2bit,
                packed_l4=packed.level4_2bit,
                radii=encoding.polar.radii,
                centroids_l1=codebooks.centroids[0],
                centroids_l2=codebooks.centroids[1],
                centroids_l3=codebooks.centroids[2],
                centroids_l4=codebooks.centroids[3],
            )

        # q_work is [B,H,T,D] and encoding chunk also has T keys.
        # Direct estimator expects per-token aligned inner products:
        #   score[b,h,t] = <q[b,h,t], k[b,h,t]>
        # so extract diagonal from [B,H,T,T].
        with nvtx_range("tq_polar_stage1_score_diag"):
            polar_dot = torch.diagonal(
                polar_dot_bhqt,
                dim1=-2,
                dim2=-1,
            )

        q_flat, prefix_shape, _ = _flatten_last_dim(
            q_work,
        )

        with nvtx_range("tq_qjl_estimate_call"):
            residual_correction_flat = qjl_inner_product_estimate(
                q=q_flat,
                encoded_r=encoding.qjl_residual,
                S=sketch,
            )

        residual_correction = residual_correction_flat.reshape(
            prefix_shape,
        )

        return polar_dot + residual_correction

    # ============================================================
    # Path B: generic reference path
    #   q: [N,D] or arbitrary [...,D]
    #
    # Used by synthetic math-quality tests.
    # ============================================================

    with nvtx_range("tq_polar_reconstruct_reference_call"):
        x_hat_polar = turboquant_polar_prod_reconstruction(
            encoding=encoding,
            codebooks=codebooks,
        ).to(torch.float32)

    if tuple(q_work.shape) != tuple(x_hat_polar.shape):
        raise ValueError(
            "q and reconstructed polar tensor must have identical shape. "
            f"Got q={tuple(q_work.shape)}, "
            f"x_hat_polar={tuple(x_hat_polar.shape)}"
        )

    with nvtx_range("tq_polar_dot_reference"):
        polar_dot = torch.sum(
            q_work * x_hat_polar,
            dim=-1,
        )

    q_flat, prefix_shape, _ = _flatten_last_dim(
        q_work,
    )

    with nvtx_range("tq_qjl_estimate_reference_call"):
        residual_correction_flat = qjl_inner_product_estimate(
            q=q_flat,
            encoded_r=encoding.qjl_residual,
            S=sketch,
        )

    residual_correction = residual_correction_flat.reshape(
        prefix_shape,
    )

    return polar_dot + residual_correction

@torch.no_grad()
def turboquant_polar_prod_score_against_chunk(
    *,
    q: torch.Tensor,
    encoding: PolarProdEncoding,
    codebooks: PolarAngleCodebooks,
    sketch: torch.Tensor,
) -> torch.Tensor:
    """
    Score query states against one compressed K chunk.

    Args:
        q:
            [B,H,Q,D]

        encoding:
            Compressed chunk whose K states correspond to:
            [B,H,T,D]

    Returns:
        scores:
            [B,H,Q,T]

    This avoids the old transitional path:
        q.expand(T) -> [B,H,T,D]
        fused CUDA score -> [B,H,T,T]
        diagonal extraction

    Stage-1 Polar score now runs directly as:
        [B,H,Q,D] x packed-[B,H,T,*] -> [B,H,Q,T]

    QJL correction remains reference-style for now.
    """
    if q.ndim != 4:
        raise ValueError(
            f"q must be [B,H,Q,D], got {tuple(q.shape)}."
        )

    q_work = q.to(torch.float32)

    B, H, Q, D = q_work.shape
    T = int(encoding.polar.radii.shape[2])

    packed = encoding.polar.packed_angles

    with nvtx_range("tq_qjl_project_q_pairwise"):
        q_projected = torch.matmul(
            q_work.to(torch.float32),
            sketch.T.to(torch.float32),
        )

    B_enc, H_enc, T_enc = encoding.polar.radii.shape[:3]

    if int(B_enc) != int(B) or int(H_enc) != int(H):
        raise ValueError(
            "QJL chunk B/H mismatch. "
            f"q has B,H=({B},{H}), "
            f"encoding has B,H=({B_enc},{H_enc})."
        )

    packed_qjl_signs = encoding.qjl_residual.packed_sign_bits.reshape(
        B,
        H,
        T,
        -1,
    )

    qjl_norms = encoding.qjl_residual.norms.reshape(
        B,
        H,
        T,
    )

    with nvtx_range("tq_fused_logits_cuda_pairwise"):
        fused_scores = turboquant_fused_logits_cuda(
            q=q_work,
            q_projected=q_projected,

            packed_l1=packed.level1_4bit,
            packed_l2=packed.level2_2bit,
            packed_l3=packed.level3_2bit,
            packed_l4=packed.level4_2bit,
            radii=encoding.polar.radii,

            centroids_l1=codebooks.centroids[0],
            centroids_l2=codebooks.centroids[1],
            centroids_l3=codebooks.centroids[2],
            centroids_l4=codebooks.centroids[3],

            packed_qjl_signs=packed_qjl_signs,
            qjl_norms=qjl_norms,
        )

    return fused_scores