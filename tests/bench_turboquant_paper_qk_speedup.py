from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import math
import time
from typing import Any

import torch

from turboquant.key_cache import TurboQuantKeyCache
from turboquant.mse_quant import (
    make_random_rotation,
    get_2bit_centroids,
)
from turboquant.qjl import (
    make_gaussian_sketch,
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

from contextlib import contextmanager

@contextmanager
def nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield

@torch.no_grad()
def cuda_time_ms(fn, warmup: int, iters: int) -> float:
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
    Build a TurboQuantKeyCache from synthetic fp32 K vectors.

    Returns:
      cache
      dense_k_fp16: [B,H,T,D], retained for einsum baseline
    """
    rotation = make_random_rotation(
        d=D,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    centroids = get_2bit_centroids(
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
        centroids=centroids,
        sketch=sketch,
        max_cache_len=T,
    )

    # Dense baseline uses fp16 K, which is more realistic for LLM inference.
    dense_k_fp16 = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=dtype,
    )

    # Runtime compressed cache currently expects key_states in the model-style shape.
    # Quantization internals cast as needed.
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
        centroids,
        sketch,
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
        centroids,
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
    
    layer = cache.layers[0]

    packed_mse = layer.packed_mse_indices_buffer[
        :, :, :T, :
    ]
    packed_qjl = layer.packed_qjl_sign_bits_buffer[
        :, :, :T, :
    ]

    packed_mse_t = packed_mse.permute(
        0, 1, 3, 2
    ).contiguous()

    packed_qjl_t = packed_qjl.permute(
        0, 1, 3, 2
    ).contiguous()

    mse_norms = layer.mse_norms_buffer[
        :, :, :T
    ].contiguous()

    qjl_norms = layer.qjl_residual_norms_buffer[
        :, :, :T
    ].contiguous()

    # ------------------------------------------------------------
    # Dense PyTorch einsum baseline
    #
    # einsum shape:
    #   q:       [B,H,1,D]
    #   dense_k: [B,H,T,D]
    #   scores:  [B,H,1,T]
    # ------------------------------------------------------------
    def dense_einsum():
        return torch.einsum(
            "bhqd,bhkd->bhqk",
            q,
            dense_k,
        )

    # ------------------------------------------------------------
    # TurboQuant packed score
    # ------------------------------------------------------------
    def tq_score():
        return turboquant_decode_score_cuda_from_cache(
            query_states=q_fp32,
            cache=cache,
            layer_idx=0,
        )

    def tq_score_transposed():
        q_flat = q_fp32.squeeze(2)

        q_rot = q_flat @ rotation.T
        q_sketch = q_flat @ sketch.T

        return turboquant_decode_score_transposed_cuda(
            q_rot=q_rot,
            q_sketch=q_sketch,
            packed_mse_indices_t=packed_mse_t,
            mse_norms=mse_norms,
            packed_qjl_sign_bits_t=packed_qjl_t,
            qjl_residual_norms=qjl_norms,
            centroids=centroids,
        )
    
    def tq_score_transposed_sharedq():
        q_flat = q_fp32.squeeze(2)

        q_rot = q_flat @ rotation.T
        q_sketch = q_flat @ sketch.T

        return turboquant_decode_score_transposed_sharedq_cuda(
            q_rot=q_rot,
            q_sketch=q_sketch,
            packed_mse_indices_t=packed_mse_t,
            mse_norms=mse_norms,
            packed_qjl_sign_bits_t=packed_qjl_t,
            qjl_residual_norms=qjl_norms,
            centroids=centroids,
        )
        
    def tq_score_mse_lut():
        q_flat = q_fp32.squeeze(2)
        q_rot = q_flat @ rotation.T

        return turboquant_mse_lut_score_transposed_cuda(
            q_rot=q_rot,
            packed_mse_indices_t=packed_mse_t,
            mse_norms=mse_norms,
            centroids=centroids,
        )

    with nvtx_range(f"paper_dense_einsum_T{T}"):
        dense_ms = cuda_time_ms(
            dense_einsum,
            warmup=warmup,
            iters=iters,
        )

    with nvtx_range(f"paper_tq_score_T{T}"):
        tq_ms = cuda_time_ms(
            tq_score,
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
        tq_mse_lut_ms = cuda_time_ms(
            tq_score_mse_lut,
            warmup=warmup,
            iters=iters,
        )

    speedup_current = dense_ms / tq_ms
    speedup_transposed = dense_ms / tq_transposed_ms
    speedup_transposed_sharedq = dense_ms / tq_transposed_sharedq_ms
    speedup_mse_lut = dense_ms / tq_mse_lut_ms

    transposed_over_current = tq_ms / tq_transposed_ms
    sharedq_over_current = tq_ms / tq_transposed_sharedq_ms
    sharedq_over_transposed = tq_transposed_ms / tq_transposed_sharedq_ms
    mse_lut_over_current = tq_ms / tq_mse_lut_ms
    mse_lut_over_sharedq = tq_transposed_sharedq_ms / tq_mse_lut_ms

    layer_report = cache.report()["layers"][0]

    dense_k_bytes = dense_k.numel() * dense_k.element_size()
    compressed_k_bytes = int(layer_report["actual_storage_bytes"])

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
            "turboquant_current_score_ms": float(tq_ms),
            "turboquant_transposed_score_ms": float(tq_transposed_ms),
            "turboquant_transposed_sharedq_score_ms": float(
                tq_transposed_sharedq_ms
            ),
            "turboquant_mse_lut_score_ms": float(tq_mse_lut_ms),
        },
        "speedup_over_einsum": {
            "current_packed_layout": float(speedup_current),
            "transposed_packed_layout": float(speedup_transposed),
            "transposed_sharedq_packed_layout": float(
                speedup_transposed_sharedq
            ),
            "mse_lut_packed_layout": float(speedup_mse_lut),
        },
        "layout_speedup": {
            "transposed_over_current": float(transposed_over_current),
            "sharedq_over_current": float(sharedq_over_current),
            "sharedq_over_transposed": float(sharedq_over_transposed),
            "mse_lut_over_current": float(mse_lut_over_current),
            "mse_lut_over_sharedq": float(mse_lut_over_sharedq),
        },
        "memory_bytes": {
            "dense_fp16_k_bytes": int(dense_k_bytes),
            "compressed_tq_k_bytes": int(compressed_k_bytes),
        },
        "memory_gb": {
            "dense_fp16_k_gb": bytes_to_gb(dense_k_bytes),
            "compressed_tq_k_gb": bytes_to_gb(compressed_k_bytes),
        },
        "compressed_over_dense_k_ratio": float(
            compressed_k_bytes / dense_k_bytes
        ),
        "cache_report": layer_report,
    }

    print(
        f"dense einsum:              {dense_ms:.6f} ms"
    )
    print(
        f"TQ current packed:         {tq_ms:.6f} ms"
    )
    print(
        f"TQ transposed packed:      {tq_transposed_ms:.6f} ms"
    )
    print(
        f"TQ transposed shared-q:    {tq_transposed_sharedq_ms:.6f} ms"
    )
    print(
        f"TQ 2-bit MSE LUT:          {tq_mse_lut_ms:.6f} ms"
    )
    print(
        f"speedup current:           {speedup_current:.4f}x"
    )
    print(
        f"speedup transposed:        {speedup_transposed:.4f}x"
    )
    print(
        f"speedup shared-q:          {speedup_transposed_sharedq:.4f}x"
    )
    print(
        f"speedup MSE LUT:           {speedup_mse_lut:.4f}x"
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
        f"MSE LUT/current:           {mse_lut_over_current:.4f}x"
    )
    print(
        f"MSE LUT/shared-q:          {mse_lut_over_sharedq:.4f}x"
    )
    print(
        f"dense K bytes:  {dense_k_bytes:,}"
    )
    print(
        f"TQ K bytes:     {compressed_k_bytes:,}"
    )
    print(
        f"K ratio:        {compressed_k_bytes / dense_k_bytes:.6f}"
    )

    # Free memory aggressively before next T.
    del cache
    del dense_k
    del q
    del q_fp32
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

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--qjl_m", type=int, default=256)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)

    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[16384, 32768, 65536, 131072],
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

    results = []

    print("========== TurboQuant paper-aligned QK^T speedup benchmark ==========")
    print(f"device       = {args.device}")
    print(f"B            = {args.batch_size}")
    print(f"H            = {args.num_heads}")
    print(f"D            = {args.head_dim}")
    print(f"M            = {args.qjl_m}")
    print(f"warmup       = {args.warmup}")
    print(f"iters        = {args.iters}")
    print(f"seq_lens     = {args.seq_lens}")
    print(f"build_chunk  = {args.build_chunk_size}")

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
            "Matches the paper's Figure 2(c) comparison style: "
            "compressed-domain QK^T score computation versus PyTorch einsum. "
            "This implementation evaluates the current project's 3-bit-style "
            "TurboQuant_prod score path, not the paper's plotted 1/2/4-bit variants."
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
