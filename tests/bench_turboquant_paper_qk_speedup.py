from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
from typing import Any

import torch

from turboquant.key_cache import TurboQuantKeyCache
from turboquant.mse_quant import (
    make_random_rotation,
    get_1bit_centroids,
    get_2bit_centroids,
    get_4bit_centroids,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)
from turboquant.cuda_packing import (
    pack_sign_bits_cuda,
)
from turboquant.cuda_score import (
    turboquant_decode_score_cuda_from_cache,
)
from turboquant.cuda_score_transposed import (
    turboquant_decode_score_transposed_cuda,
    turboquant_decode_score_transposed_sharedq_cuda,
)
from turboquant.cuda_score_mse_lut import (
    turboquant_mse_lut_score_transposed_cuda,
)
from turboquant.cuda_score_mse_lut_1bit import (
    turboquant_mse_lut_1bit_score_transposed_cuda,
)
from turboquant.cuda_score_mse_lut_4bit import (
    turboquant_mse_lut_4bit_score_transposed_cuda,
)


@torch.no_grad()
def cuda_time_ms(
    fn,
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        _ = fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        _ = fn()

    end.record()

    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return float(total_ms / iters)


def bytes_to_gb(x: int) -> float:
    return float(x) / (1024 ** 3)


@torch.no_grad()
def build_compressed_cache(
    *,
    B: int,
    H: int,
    T: int,
    D: int,
    M: int,
    device: str,
    dtype: torch.dtype,
    chunk_size: int,
) -> tuple[
    TurboQuantKeyCache,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build:
      - dense fp16 K for dense einsum baseline
      - current TurboQuant_prod compressed cache
      - rotation / 2-bit centroids / sketch
    """

    rotation = make_random_rotation(
        d=D,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    centroids_2bit = get_2bit_centroids(
        d=D,
        device=device,
        dtype=torch.float32,
    )

    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=456,
    )

    cache = TurboQuantKeyCache(
        num_layers=1,
        rotation=rotation,
        centroids=centroids_2bit,
        sketch=sketch,
        max_cache_len=T,
    )

    dense_k_fp16 = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=dtype,
    )

    for begin in range(0, T, chunk_size):
        end = min(begin + chunk_size, T)

        key_chunk = dense_k_fp16[:, :, begin:end, :]

        cache.append(
            layer_idx=0,
            key_states=key_chunk,
            value_states=None,
        )

    return (
        cache,
        dense_k_fp16,
        rotation,
        centroids_2bit,
        sketch,
    )


@torch.no_grad()
def build_1bit_mse_cache_tensors(
    *,
    dense_k: torch.Tensor,
    rotation: torch.Tensor,
    D: int,
    device: str,
    chunk_size: int = 1024,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build paper-benchmark-only 1-bit MSE tensors in chunks.

    Returns:
      packed_1bit_t:   [B,H,D/8,T]
      mse_norms_1bit: [B,H,T]
      centroids_1bit: [2]

    Chunking avoids materializing full-size:
      k_rot  [B,H,T,D]
      k_norm [B,H,T,D]
    for very long sequence lengths.
    """

    B, H, T, _ = dense_k.shape
    packed_D = D // 8

    centroids_1bit = get_1bit_centroids(
        d=D,
        device=device,
        dtype=torch.float32,
    )

    packed_1bit_t = torch.empty(
        B,
        H,
        packed_D,
        T,
        device=device,
        dtype=torch.uint8,
    )

    mse_norms_1bit = torch.empty(
        B,
        H,
        T,
        device=device,
        dtype=torch.float32,
    )

    for begin in range(0, T, chunk_size):
        end = min(begin + chunk_size, T)
        t_chunk = end - begin

        k_chunk = dense_k[:, :, begin:end, :].to(torch.float32)

        k_flat = k_chunk.reshape(
            B * H * t_chunk,
            D,
        )

        k_rot = k_flat @ rotation.T

        norms_flat = torch.linalg.vector_norm(
            k_rot,
            ord=2,
            dim=-1,
        )

        safe_norms = torch.clamp(
            norms_flat,
            min=torch.finfo(k_rot.dtype).eps,
        )

        k_norm = k_rot / safe_norms.unsqueeze(-1)

        k_norm_bhtd = k_norm.reshape(
            B,
            H,
            t_chunk,
            D,
        ).contiguous()

        packed_chunk = pack_sign_bits_cuda(
            k_norm_bhtd
        )

        packed_1bit_t[:, :, :, begin:end].copy_(
            packed_chunk.permute(
                0,
                1,
                3,
                2,
            ).contiguous()
        )

        mse_norms_1bit[:, :, begin:end].copy_(
            norms_flat.reshape(
                B,
                H,
                t_chunk,
            )
        )

        del k_chunk
        del k_flat
        del k_rot
        del norms_flat
        del safe_norms
        del k_norm
        del k_norm_bhtd
        del packed_chunk

    return (
        packed_1bit_t,
        mse_norms_1bit,
        centroids_1bit,
    )


@torch.no_grad()
def build_4bit_mse_cache_tensors(
    *,
    dense_k: torch.Tensor,
    rotation: torch.Tensor,
    D: int,
    device: str,
    chunk_size: int = 1024,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build paper-benchmark-only 4-bit MSE tensors in chunks.

    Returns:
      packed_4bit_t:   [B,H,D/2,T]
      mse_norms_4bit: [B,H,T]
      centroids_4bit: [16]

    One uint8 stores two 4-bit centroid indices.
    """

    B, H, T, _ = dense_k.shape
    packed_D = D // 2

    centroids_4bit = get_4bit_centroids(
        d=D,
        device=device,
        dtype=torch.float32,
    )

    packed_4bit_t = torch.empty(
        B,
        H,
        packed_D,
        T,
        device=device,
        dtype=torch.uint8,
    )

    mse_norms_4bit = torch.empty(
        B,
        H,
        T,
        device=device,
        dtype=torch.float32,
    )

    for begin in range(0, T, chunk_size):
        end = min(begin + chunk_size, T)
        t_chunk = end - begin

        k_chunk = dense_k[:, :, begin:end, :].to(torch.float32)

        k_flat = k_chunk.reshape(
            B * H * t_chunk,
            D,
        )

        k_rot = k_flat @ rotation.T

        norms_flat = torch.linalg.vector_norm(
            k_rot,
            ord=2,
            dim=-1,
        )

        safe_norms = torch.clamp(
            norms_flat,
            min=torch.finfo(k_rot.dtype).eps,
        )

        k_norm = k_rot / safe_norms.unsqueeze(-1)

        k_norm_bhtd = k_norm.reshape(
            B,
            H,
            t_chunk,
            D,
        ).contiguous()

        dist = torch.abs(
            k_norm_bhtd.unsqueeze(-1) -
            centroids_4bit.view(1, 1, 1, 1, 16)
        )

        indices = torch.argmin(
            dist,
            dim=-1,
        )

        low = indices[..., 0::2].to(torch.uint8)
        high = indices[..., 1::2].to(torch.uint8)

        packed_chunk = low | (high << 4)

        packed_4bit_t[:, :, :, begin:end].copy_(
            packed_chunk.permute(
                0,
                1,
                3,
                2,
            ).contiguous()
        )

        mse_norms_4bit[:, :, begin:end].copy_(
            norms_flat.reshape(
                B,
                H,
                t_chunk,
            )
        )

        del k_chunk
        del k_flat
        del k_rot
        del norms_flat
        del safe_norms
        del k_norm
        del k_norm_bhtd
        del dist
        del indices
        del low
        del high
        del packed_chunk

    return (
        packed_4bit_t,
        mse_norms_4bit,
        centroids_4bit,
    )


@torch.no_grad()
def benchmark_one_length(
    *,
    T: int,
    B: int,
    H: int,
    D: int,
    M: int,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    build_chunk_size: int,
) -> dict[str, Any]:
    print()
    print("============================================================")
    print(f"[Benchmark] T={T}")
    print("============================================================")

    (
        cache,
        dense_k,
        rotation,
        centroids_2bit,
        sketch,
    ) = build_compressed_cache(
        B=B,
        H=H,
        T=T,
        D=D,
        M=M,
        device=device,
        dtype=dtype,
        chunk_size=build_chunk_size,
    )

    q = torch.randn(
        B,
        H,
        1,
        D,
        device=device,
        dtype=dtype,
    )

    q_fp32 = q.to(torch.float32)

    # ============================================================
    # Existing TurboQuant_prod packed cache tensors
    # ============================================================

    layer = cache.layers[0]

    packed_mse_2bit = layer.packed_mse_indices_buffer[
        :, :, :T, :
    ]

    packed_qjl = layer.packed_qjl_sign_bits_buffer[
        :, :, :T, :
    ]

    packed_mse_2bit_t = packed_mse_2bit.permute(
        0,
        1,
        3,
        2,
    ).contiguous()

    packed_qjl_t = packed_qjl.permute(
        0,
        1,
        3,
        2,
    ).contiguous()

    mse_norms_2bit = layer.mse_norms_buffer[
        :, :, :T
    ].contiguous()

    qjl_norms = layer.qjl_residual_norms_buffer[
        :, :, :T
    ].contiguous()

    # ============================================================
    # Benchmark-only 1-bit MSE tensors
    # ============================================================

    (
        packed_1bit_t,
        mse_norms_1bit,
        centroids_1bit,
    ) = build_1bit_mse_cache_tensors(
        dense_k=dense_k,
        rotation=rotation,
        D=D,
        device=device,
        chunk_size=build_chunk_size,
    )

    # ============================================================
    # Benchmark-only 4-bit MSE tensors
    # ============================================================

    (
        packed_4bit_t,
        mse_norms_4bit,
        centroids_4bit,
    ) = build_4bit_mse_cache_tensors(
        dense_k=dense_k,
        rotation=rotation,
        D=D,
        device=device,
        chunk_size=build_chunk_size,
    )

    # ============================================================
    # Dense fp16 baseline
    # ============================================================

    def dense_einsum():
        return torch.einsum(
            "bhqd,bhkd->bhqk",
            q,
            dense_k,
        )

    # ============================================================
    # Current TurboQuant_prod packed score
    # ============================================================

    def tq_score_current():
        return turboquant_decode_score_cuda_from_cache(
            query_states=q_fp32,
            cache=cache,
            layer_idx=0,
        )

    # ============================================================
    # TurboQuant_prod transposed packed score
    # ============================================================

    def tq_score_transposed():
        q_flat = q_fp32.squeeze(2)

        q_rot = q_flat @ rotation.T
        q_sketch = q_flat @ sketch.T

        return turboquant_decode_score_transposed_cuda(
            q_rot=q_rot,
            q_sketch=q_sketch,
            packed_mse_indices_t=packed_mse_2bit_t,
            mse_norms=mse_norms_2bit,
            packed_qjl_sign_bits_t=packed_qjl_t,
            qjl_residual_norms=qjl_norms,
            centroids=centroids_2bit,
        )

    # ============================================================
    # TurboQuant_prod transposed + shared-q score
    # ============================================================

    def tq_score_transposed_sharedq():
        q_flat = q_fp32.squeeze(2)

        q_rot = q_flat @ rotation.T
        q_sketch = q_flat @ sketch.T

        return turboquant_decode_score_transposed_sharedq_cuda(
            q_rot=q_rot,
            q_sketch=q_sketch,
            packed_mse_indices_t=packed_mse_2bit_t,
            mse_norms=mse_norms_2bit,
            packed_qjl_sign_bits_t=packed_qjl_t,
            qjl_residual_norms=qjl_norms,
            centroids=centroids_2bit,
        )

    # ============================================================
    # 1-bit MSE LUT fused score
    # ============================================================

    def tq_score_mse_lut_1bit():
        q_flat = q_fp32.squeeze(2)
        q_rot = q_flat @ rotation.T

        return turboquant_mse_lut_1bit_score_transposed_cuda(
            q_rot=q_rot,
            packed_mse_sign_bits_t=packed_1bit_t,
            mse_norms=mse_norms_1bit,
            centroids=centroids_1bit,
        )

    # ============================================================
    # 2-bit MSE LUT fused score
    # ============================================================

    def tq_score_mse_lut_2bit():
        q_flat = q_fp32.squeeze(2)
        q_rot = q_flat @ rotation.T

        return turboquant_mse_lut_score_transposed_cuda(
            q_rot=q_rot,
            packed_mse_indices_t=packed_mse_2bit_t,
            mse_norms=mse_norms_2bit,
            centroids=centroids_2bit,
        )

    # ============================================================
    # 4-bit MSE LUT fused score
    # ============================================================

    def tq_score_mse_lut_4bit():
        q_flat = q_fp32.squeeze(2)
        q_rot = q_flat @ rotation.T

        return turboquant_mse_lut_4bit_score_transposed_cuda(
            q_rot=q_rot,
            packed_indices_t=packed_4bit_t,
            mse_norms=mse_norms_4bit,
            centroids=centroids_4bit,
        )

    # ============================================================
    # Timing
    # ============================================================

    dense_ms = cuda_time_ms(
        dense_einsum,
        warmup=warmup,
        iters=iters,
    )

    tq_current_ms = cuda_time_ms(
        tq_score_current,
        warmup=warmup,
        iters=iters,
    )

    tq_transposed_ms = cuda_time_ms(
        tq_score_transposed,
        warmup=warmup,
        iters=iters,
    )

    tq_transposed_sharedq_ms = cuda_time_ms(
        tq_score_transposed_sharedq,
        warmup=warmup,
        iters=iters,
    )

    tq_mse_lut_1bit_ms = cuda_time_ms(
        tq_score_mse_lut_1bit,
        warmup=warmup,
        iters=iters,
    )

    tq_mse_lut_2bit_ms = cuda_time_ms(
        tq_score_mse_lut_2bit,
        warmup=warmup,
        iters=iters,
    )

    tq_mse_lut_4bit_ms = cuda_time_ms(
        tq_score_mse_lut_4bit,
        warmup=warmup,
        iters=iters,
    )

    # ============================================================
    # Speedups over dense einsum
    # ============================================================

    speedup_current = dense_ms / tq_current_ms
    speedup_transposed = dense_ms / tq_transposed_ms
    speedup_sharedq = dense_ms / tq_transposed_sharedq_ms

    speedup_mse_lut_1bit = dense_ms / tq_mse_lut_1bit_ms
    speedup_mse_lut_2bit = dense_ms / tq_mse_lut_2bit_ms
    speedup_mse_lut_4bit = dense_ms / tq_mse_lut_4bit_ms

    # ============================================================
    # Relative speedups
    # ============================================================

    transposed_over_current = tq_current_ms / tq_transposed_ms
    sharedq_over_current = tq_current_ms / tq_transposed_sharedq_ms
    sharedq_over_transposed = tq_transposed_ms / tq_transposed_sharedq_ms

    mse_lut_1bit_over_current = tq_current_ms / tq_mse_lut_1bit_ms
    mse_lut_1bit_over_sharedq = tq_transposed_sharedq_ms / tq_mse_lut_1bit_ms

    mse_lut_2bit_over_current = tq_current_ms / tq_mse_lut_2bit_ms
    mse_lut_2bit_over_sharedq = tq_transposed_sharedq_ms / tq_mse_lut_2bit_ms

    mse_lut_4bit_over_current = tq_current_ms / tq_mse_lut_4bit_ms
    mse_lut_4bit_over_sharedq = tq_transposed_sharedq_ms / tq_mse_lut_4bit_ms

    mse_lut_1bit_over_2bit = tq_mse_lut_2bit_ms / tq_mse_lut_1bit_ms
    mse_lut_2bit_over_4bit = tq_mse_lut_4bit_ms / tq_mse_lut_2bit_ms
    mse_lut_1bit_over_4bit = tq_mse_lut_4bit_ms / tq_mse_lut_1bit_ms

    # ============================================================
    # Memory accounting
    # ============================================================

    layer_report = cache.report()["layers"][0]

    dense_k_bytes = (
        dense_k.numel() *
        dense_k.element_size()
    )

    compressed_prod_k_bytes = int(
        layer_report["actual_storage_bytes"]
    )

    packed_1bit_bytes = (
        packed_1bit_t.numel() *
        packed_1bit_t.element_size()
    )

    mse_norms_1bit_bytes = (
        mse_norms_1bit.numel() *
        mse_norms_1bit.element_size()
    )

    mse_1bit_total_bytes = (
        packed_1bit_bytes +
        mse_norms_1bit_bytes
    )

    packed_4bit_bytes = (
        packed_4bit_t.numel() *
        packed_4bit_t.element_size()
    )

    mse_norms_4bit_bytes = (
        mse_norms_4bit.numel() *
        mse_norms_4bit.element_size()
    )

    mse_4bit_total_bytes = (
        packed_4bit_bytes +
        mse_norms_4bit_bytes
    )

    # ============================================================
    # Console output
    # ============================================================

    print(
        f"dense einsum:              {dense_ms:.6f} ms"
    )
    print(
        f"TQ current packed:         {tq_current_ms:.6f} ms"
    )
    print(
        f"TQ transposed packed:      {tq_transposed_ms:.6f} ms"
    )
    print(
        f"TQ transposed shared-q:    {tq_transposed_sharedq_ms:.6f} ms"
    )
    print(
        f"TQ 1-bit MSE LUT:          {tq_mse_lut_1bit_ms:.6f} ms"
    )
    print(
        f"TQ 2-bit MSE LUT:          {tq_mse_lut_2bit_ms:.6f} ms"
    )
    print(
        f"TQ 4-bit MSE LUT:          {tq_mse_lut_4bit_ms:.6f} ms"
    )

    print(
        f"speedup current:           {speedup_current:.4f}x"
    )
    print(
        f"speedup transposed:        {speedup_transposed:.4f}x"
    )
    print(
        f"speedup shared-q:          {speedup_sharedq:.4f}x"
    )
    print(
        f"speedup 1-bit MSE LUT:     {speedup_mse_lut_1bit:.4f}x"
    )
    print(
        f"speedup 2-bit MSE LUT:     {speedup_mse_lut_2bit:.4f}x"
    )
    print(
        f"speedup 4-bit MSE LUT:     {speedup_mse_lut_4bit:.4f}x"
    )

    print(
        f"transposed/current:        {transposed_over_current:.4f}x"
    )
    print(
        f"shared-q/current:          {sharedq_over_current:.4f}x"
    )
    print(
        f"shared-q/transposed:       {sharedq_over_transposed:.4f}x"
    )

    print(
        f"1-bit LUT/current:         {mse_lut_1bit_over_current:.4f}x"
    )
    print(
        f"1-bit LUT/shared-q:        {mse_lut_1bit_over_sharedq:.4f}x"
    )

    print(
        f"2-bit LUT/current:         {mse_lut_2bit_over_current:.4f}x"
    )
    print(
        f"2-bit LUT/shared-q:        {mse_lut_2bit_over_sharedq:.4f}x"
    )

    print(
        f"4-bit LUT/current:         {mse_lut_4bit_over_current:.4f}x"
    )
    print(
        f"4-bit LUT/shared-q:        {mse_lut_4bit_over_sharedq:.4f}x"
    )

    print(
        f"1-bit LUT/2-bit LUT:       {mse_lut_1bit_over_2bit:.4f}x"
    )
    print(
        f"2-bit LUT/4-bit LUT:       {mse_lut_2bit_over_4bit:.4f}x"
    )
    print(
        f"1-bit LUT/4-bit LUT:       {mse_lut_1bit_over_4bit:.4f}x"
    )

    print(
        f"dense K bytes:   {dense_k_bytes:,}"
    )
    print(
        f"TQ prod K bytes: {compressed_prod_k_bytes:,}"
    )
    print(
        f"Prod K ratio:    {compressed_prod_k_bytes / dense_k_bytes:.6f}"
    )
    print(
        f"1-bit MSE bytes: {mse_1bit_total_bytes:,}"
    )
    print(
        f"1-bit K ratio:   {mse_1bit_total_bytes / dense_k_bytes:.6f}"
    )
    print(
        f"4-bit MSE bytes: {mse_4bit_total_bytes:,}"
    )
    print(
        f"4-bit K ratio:   {mse_4bit_total_bytes / dense_k_bytes:.6f}"
    )

    # ============================================================
    # JSON result
    # ============================================================

    result = {
        "T": int(T),
        "shape": {
            "B": int(B),
            "H": int(H),
            "Q": 1,
            "D": int(D),
            "M": int(M),
        },
        "timing_ms": {
            "dense_einsum_ms": float(dense_ms),
            "turboquant_current_score_ms": float(tq_current_ms),
            "turboquant_transposed_score_ms": float(tq_transposed_ms),
            "turboquant_transposed_sharedq_score_ms": float(
                tq_transposed_sharedq_ms
            ),
            "turboquant_mse_lut_1bit_score_ms": float(
                tq_mse_lut_1bit_ms
            ),
            "turboquant_mse_lut_2bit_score_ms": float(
                tq_mse_lut_2bit_ms
            ),
            "turboquant_mse_lut_4bit_score_ms": float(
                tq_mse_lut_4bit_ms
            ),
        },
        "speedup_over_einsum": {
            "current_packed_layout": float(speedup_current),
            "transposed_packed_layout": float(speedup_transposed),
            "transposed_sharedq_packed_layout": float(speedup_sharedq),
            "mse_lut_1bit_packed_layout": float(speedup_mse_lut_1bit),
            "mse_lut_2bit_packed_layout": float(speedup_mse_lut_2bit),
            "mse_lut_4bit_packed_layout": float(speedup_mse_lut_4bit),
        },
        "layout_speedup": {
            "transposed_over_current": float(transposed_over_current),
            "sharedq_over_current": float(sharedq_over_current),
            "sharedq_over_transposed": float(sharedq_over_transposed),

            "mse_lut_1bit_over_current": float(
                mse_lut_1bit_over_current
            ),
            "mse_lut_1bit_over_sharedq": float(
                mse_lut_1bit_over_sharedq
            ),

            "mse_lut_2bit_over_current": float(
                mse_lut_2bit_over_current
            ),
            "mse_lut_2bit_over_sharedq": float(
                mse_lut_2bit_over_sharedq
            ),

            "mse_lut_4bit_over_current": float(
                mse_lut_4bit_over_current
            ),
            "mse_lut_4bit_over_sharedq": float(
                mse_lut_4bit_over_sharedq
            ),

            "mse_lut_1bit_over_2bit": float(
                mse_lut_1bit_over_2bit
            ),
            "mse_lut_2bit_over_4bit": float(
                mse_lut_2bit_over_4bit
            ),
            "mse_lut_1bit_over_4bit": float(
                mse_lut_1bit_over_4bit
            ),
        },
        "memory_bytes": {
            "dense_fp16_k_bytes": int(dense_k_bytes),
            "compressed_prod_k_bytes": int(compressed_prod_k_bytes),

            "mse_1bit_packed_bytes": int(packed_1bit_bytes),
            "mse_1bit_norm_bytes": int(mse_norms_1bit_bytes),
            "mse_1bit_total_bytes": int(mse_1bit_total_bytes),

            "mse_4bit_packed_bytes": int(packed_4bit_bytes),
            "mse_4bit_norm_bytes": int(mse_norms_4bit_bytes),
            "mse_4bit_total_bytes": int(mse_4bit_total_bytes),
        },
        "memory_gb": {
            "dense_fp16_k_gb": bytes_to_gb(dense_k_bytes),
            "compressed_prod_k_gb": bytes_to_gb(compressed_prod_k_bytes),
            "mse_1bit_total_gb": bytes_to_gb(mse_1bit_total_bytes),
            "mse_4bit_total_gb": bytes_to_gb(mse_4bit_total_bytes),
        },
        "memory_ratio": {
            "prod_k_over_dense_fp16_k": float(
                compressed_prod_k_bytes / dense_k_bytes
            ),
            "mse_1bit_over_dense_fp16_k": float(
                mse_1bit_total_bytes / dense_k_bytes
            ),
            "mse_4bit_over_dense_fp16_k": float(
                mse_4bit_total_bytes / dense_k_bytes
            ),
        },
        "prod_cache_report": layer_report,
    }

    # ============================================================
    # Cleanup
    # ============================================================

    del cache
    del dense_k
    del q
    del q_fp32

    del packed_mse_2bit
    del packed_qjl
    del packed_mse_2bit_t
    del packed_qjl_t
    del mse_norms_2bit
    del qjl_norms

    del packed_1bit_t
    del mse_norms_1bit
    del centroids_1bit

    del packed_4bit_t
    del mse_norms_4bit
    del centroids_4bit

    torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16"],
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--head_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--qjl_m",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[
            16384,
            32768,
            65536,
            131072,
        ],
    )

    parser.add_argument(
        "--build_chunk_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--out",
        type=str,
        default=(
            "runs/svd_uniform_08/eval/"
            "bench_turboquant_paper_qk_speedup.json"
        ),
    )

    args = parser.parse_args()

    dtype = torch.float16

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    print(
        "========== TurboQuant paper-aligned QK^T speedup benchmark =========="
    )
    print(f"device       = {args.device}")
    print(f"B            = {args.batch_size}")
    print(f"H            = {args.num_heads}")
    print(f"D            = {args.head_dim}")
    print(f"M            = {args.qjl_m}")
    print(f"warmup       = {args.warmup}")
    print(f"iters        = {args.iters}")
    print(f"seq_lens     = {args.seq_lens}")
    print(f"build_chunk  = {args.build_chunk_size}")

    results = []

    for T in args.seq_lens:
        result = benchmark_one_length(
            T=T,
            B=args.batch_size,
            H=args.num_heads,
            D=args.head_dim,
            M=args.qjl_m,
            device=args.device,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            build_chunk_size=args.build_chunk_size,
        )

        results.append(result)

    summary = {
        "benchmark": "paper_aligned_qk_speedup_vs_pytorch_einsum",
        "note": (
            "This benchmark compares dense fp16 einsum against: "
            "TurboQuant_prod packed score kernels, "
            "1-bit MSE LUT fused score, "
            "2-bit MSE LUT fused score, and "
            "4-bit MSE LUT fused score."
        ),
        "config": {
            "device": args.device,
            "B": args.batch_size,
            "H": args.num_heads,
            "D": args.head_dim,
            "M": args.qjl_m,
            "warmup": args.warmup,
            "iters": args.iters,
            "build_chunk_size": args.build_chunk_size,
            "seq_lens": args.seq_lens,
        },
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"[Save] {out}")
    print("[PASS] Paper-aligned QK^T speedup benchmark completed.")


if __name__ == "__main__":
    main()