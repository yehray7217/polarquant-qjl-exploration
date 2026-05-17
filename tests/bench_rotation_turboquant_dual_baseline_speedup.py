#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "tests" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from turboquant.key_cache import TurboQuantKeyCache
from turboquant.mse_quant import (
    make_random_rotation,
    get_1bit_centroids,
    get_2bit_centroids,
    get_4bit_centroids,
)
from turboquant.qjl import make_gaussian_sketch
from turboquant.cuda_packing import pack_sign_bits_cuda
from turboquant.cuda_score import turboquant_decode_score_cuda_from_cache
from turboquant.cuda_score_transposed import (
    turboquant_decode_score_transposed_cuda,
    turboquant_decode_score_transposed_sharedq_cuda,
)
from turboquant.cuda_score_mse_lut import turboquant_mse_lut_score_transposed_cuda
from turboquant.cuda_score_mse_lut_1bit import turboquant_mse_lut_1bit_score_transposed_cuda
from turboquant.cuda_score_mse_lut_4bit import turboquant_mse_lut_4bit_score_transposed_cuda


def ensure_cuda(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"Expected CUDA device, got {device!r}.")
    return dev


def sync() -> None:
    torch.cuda.synchronize()


@torch.no_grad()
def bench_cuda_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}")

    out: torch.Tensor | None = None
    for _ in range(warmup):
        out = fn()

    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    sync()

    if out is None:
        raise RuntimeError("Benchmark function did not produce output.")
    return float(start.elapsed_time(end)) / float(iters), out


def tensor_bytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())


def safe_ratio(a: float, b: float) -> float:
    return float("inf") if b == 0 else float(a / b)


@torch.no_grad()
def score_error_metrics(
    *,
    scores_ref: torch.Tensor,
    scores_test: torch.Tensor,
) -> dict[str, float]:
    ref = scores_ref.to(torch.float32)
    test = scores_test.to(torch.float32)
    if tuple(ref.shape) != tuple(test.shape):
        raise ValueError(f"Score shape mismatch: ref={tuple(ref.shape)}, test={tuple(test.shape)}")

    err = test - ref
    abs_err = err.abs()
    mae = float(abs_err.mean().item())
    rmse = float(torch.sqrt(torch.mean(err * err)).item())
    ref_abs_mean = float(ref.abs().mean().item())
    ref_rms = float(torch.sqrt(torch.mean(ref * ref)).item())

    return {
        "mae": mae,
        "rmse": rmse,
        "relative_mae": safe_ratio(mae, ref_abs_mean),
        "relative_rmse": safe_ratio(rmse, ref_rms),
        "max_abs_error": float(abs_err.max().item()),
    }


@torch.no_grad()
def build_rotation_cache(
    *,
    B: int,
    H: int,
    T: int,
    D: int,
    M: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[TurboQuantKeyCache, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation = make_random_rotation(
        d=D,
        device=str(device),
        dtype=torch.float32,
        seed=123,
    )
    centroids_2bit = get_2bit_centroids(
        d=D,
        device=str(device),
        dtype=torch.float32,
    )
    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=str(device),
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
        dtype=torch.float16,
    )

    for begin in range(0, T, chunk_size):
        end = min(begin + chunk_size, T)
        key_chunk = dense_k_fp16[:, :, begin:end, :]
        cache.append(layer_idx=0, key_states=key_chunk, value_states=None)

    return cache, dense_k_fp16, rotation, centroids_2bit, sketch


@torch.no_grad()
def build_1bit_mse_cache_tensors(
    *,
    dense_k_fp16: torch.Tensor,
    rotation: torch.Tensor,
    D: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, _ = dense_k_fp16.shape
    packed_D = D // 8

    centroids_1bit = get_1bit_centroids(
        d=D,
        device=str(device),
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

        k_chunk = dense_k_fp16[:, :, begin:end, :].to(torch.float32)
        k_flat = k_chunk.reshape(B * H * t_chunk, D)
        k_rot = k_flat @ rotation.T
        norms_flat = torch.linalg.vector_norm(k_rot, ord=2, dim=-1)
        safe_norms = torch.clamp(norms_flat, min=torch.finfo(k_rot.dtype).eps)
        k_norm = k_rot / safe_norms.unsqueeze(-1)
        k_norm_bhtd = k_norm.reshape(B, H, t_chunk, D).contiguous()

        packed_chunk = pack_sign_bits_cuda(k_norm_bhtd)
        packed_1bit_t[:, :, :, begin:end].copy_(packed_chunk.permute(0, 1, 3, 2).contiguous())
        mse_norms_1bit[:, :, begin:end].copy_(norms_flat.reshape(B, H, t_chunk))

        del k_chunk, k_flat, k_rot, norms_flat, safe_norms, k_norm, k_norm_bhtd, packed_chunk

    return packed_1bit_t, mse_norms_1bit, centroids_1bit


@torch.no_grad()
def build_4bit_mse_cache_tensors(
    *,
    dense_k_fp16: torch.Tensor,
    rotation: torch.Tensor,
    D: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, _ = dense_k_fp16.shape
    packed_D = D // 2

    centroids_4bit = get_4bit_centroids(
        d=D,
        device=str(device),
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

        k_chunk = dense_k_fp16[:, :, begin:end, :].to(torch.float32)
        k_flat = k_chunk.reshape(B * H * t_chunk, D)
        k_rot = k_flat @ rotation.T
        norms_flat = torch.linalg.vector_norm(k_rot, ord=2, dim=-1)
        safe_norms = torch.clamp(norms_flat, min=torch.finfo(k_rot.dtype).eps)
        k_norm = k_rot / safe_norms.unsqueeze(-1)
        k_norm_bhtd = k_norm.reshape(B, H, t_chunk, D).contiguous()

        dist = torch.abs(k_norm_bhtd.unsqueeze(-1) - centroids_4bit.view(1, 1, 1, 1, 16))
        indices = torch.argmin(dist, dim=-1)
        low = indices[..., 0::2].to(torch.uint8)
        high = indices[..., 1::2].to(torch.uint8)
        packed_chunk = low | (high << 4)

        packed_4bit_t[:, :, :, begin:end].copy_(packed_chunk.permute(0, 1, 3, 2).contiguous())
        mse_norms_4bit[:, :, begin:end].copy_(norms_flat.reshape(B, H, t_chunk))

        del k_chunk, k_flat, k_rot, norms_flat, safe_norms, k_norm, k_norm_bhtd
        del dist, indices, low, high, packed_chunk

    return packed_4bit_t, mse_norms_4bit, centroids_4bit


@torch.no_grad()
def benchmark_one_length(
    *,
    T: int,
    B: int,
    H: int,
    D: int,
    M: int,
    device: torch.device,
    warmup: int,
    iters: int,
    build_chunk_size: int,
) -> dict[str, Any]:
    print()
    print("=" * 72)
    print(f"[Rotation-based TurboQuant dual-baseline benchmark] T={T}")
    print("=" * 72)

    cache, dense_k_fp16, rotation, centroids_2bit, sketch = build_rotation_cache(
        B=B,
        H=H,
        T=T,
        D=D,
        M=M,
        device=device,
        chunk_size=build_chunk_size,
    )

    q_fp16 = torch.randn(B, H, 1, D, device=device, dtype=torch.float16)
    q_fp32 = q_fp16.to(torch.float32)
    dense_k_fp32 = dense_k_fp16.to(torch.float32)

    layer = cache.layers[0]
    packed_mse_2bit = layer.packed_mse_indices_buffer[:, :, :T, :]
    packed_qjl = layer.packed_qjl_sign_bits_buffer[:, :, :T, :]
    packed_mse_2bit_t = packed_mse_2bit.permute(0, 1, 3, 2).contiguous()
    packed_qjl_t = packed_qjl.permute(0, 1, 3, 2).contiguous()
    mse_norms_2bit = layer.mse_norms_buffer[:, :, :T].contiguous()
    qjl_norms = layer.qjl_residual_norms_buffer[:, :, :T].contiguous()

    packed_1bit_t, mse_norms_1bit, centroids_1bit = build_1bit_mse_cache_tensors(
        dense_k_fp16=dense_k_fp16,
        rotation=rotation,
        D=D,
        device=device,
        chunk_size=build_chunk_size,
    )
    packed_4bit_t, mse_norms_4bit, centroids_4bit = build_4bit_mse_cache_tensors(
        dense_k_fp16=dense_k_fp16,
        rotation=rotation,
        D=D,
        device=device,
        chunk_size=build_chunk_size,
    )

    def dense_fp16_einsum() -> torch.Tensor:
        return torch.einsum("bhqd,bhkd->bhqk", q_fp16, dense_k_fp16)

    def dense_fp32_qkt() -> torch.Tensor:
        return torch.matmul(q_fp32, dense_k_fp32.transpose(-1, -2))

    def tq_current() -> torch.Tensor:
        return turboquant_decode_score_cuda_from_cache(
            query_states=q_fp32,
            cache=cache,
            layer_idx=0,
        )

    def tq_transposed() -> torch.Tensor:
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

    def tq_transposed_sharedq() -> torch.Tensor:
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

    def tq_mse_lut_1bit() -> torch.Tensor:
        q_flat = q_fp32.squeeze(2)
        q_rot = q_flat @ rotation.T
        return turboquant_mse_lut_1bit_score_transposed_cuda(
            q_rot=q_rot,
            packed_mse_sign_bits_t=packed_1bit_t,
            mse_norms=mse_norms_1bit,
            centroids=centroids_1bit,
        )

    def tq_mse_lut_2bit() -> torch.Tensor:
        q_flat = q_fp32.squeeze(2)
        q_rot = q_flat @ rotation.T
        return turboquant_mse_lut_score_transposed_cuda(
            q_rot=q_rot,
            packed_mse_indices_t=packed_mse_2bit_t,
            mse_norms=mse_norms_2bit,
            centroids=centroids_2bit,
        )

    def tq_mse_lut_4bit() -> torch.Tensor:
        q_flat = q_fp32.squeeze(2)
        q_rot = q_flat @ rotation.T
        return turboquant_mse_lut_4bit_score_transposed_cuda(
            q_rot=q_rot,
            packed_indices_t=packed_4bit_t,
            mse_norms=mse_norms_4bit,
            centroids=centroids_4bit,
        )

    dense_fp16_ms, dense_fp16_scores = bench_cuda_ms(dense_fp16_einsum, warmup=warmup, iters=iters)
    dense_fp32_ms, dense_fp32_scores = bench_cuda_ms(dense_fp32_qkt, warmup=warmup, iters=iters)

    variants: dict[str, tuple[float, torch.Tensor]] = {}
    for name, fn in {
        "current_packed_layout": tq_current,
        "transposed_packed_layout": tq_transposed,
        "transposed_sharedq_packed_layout": tq_transposed_sharedq,
        "mse_lut_1bit_packed_layout": tq_mse_lut_1bit,
        "mse_lut_2bit_packed_layout": tq_mse_lut_2bit,
        "mse_lut_4bit_packed_layout": tq_mse_lut_4bit,
    }.items():
        variants[name] = bench_cuda_ms(fn, warmup=warmup, iters=iters)

    layer_report = cache.report()["layers"][0]
    compressed_prod_k_bytes = int(layer_report["actual_storage_bytes"])
    dense_fp16_k_bytes = tensor_bytes(dense_k_fp16)
    dense_fp32_k_bytes = tensor_bytes(dense_k_fp32)

    packed_1bit_bytes = tensor_bytes(packed_1bit_t)
    mse_norms_1bit_bytes = tensor_bytes(mse_norms_1bit)
    mse_1bit_total_bytes = packed_1bit_bytes + mse_norms_1bit_bytes

    packed_4bit_bytes = tensor_bytes(packed_4bit_t)
    mse_norms_4bit_bytes = tensor_bytes(mse_norms_4bit)
    mse_4bit_total_bytes = packed_4bit_bytes + mse_norms_4bit_bytes

    timing_ms = {
        "dense_fp16_einsum_ms": float(dense_fp16_ms),
        "dense_fp32_qkt_ms": float(dense_fp32_ms),
    }
    timing_ms.update({f"{name}_ms": float(ms) for name, (ms, _) in variants.items()})

    speedup_over_fp16 = {name: safe_ratio(dense_fp16_ms, ms) for name, (ms, _) in variants.items()}
    speedup_over_fp32 = {name: safe_ratio(dense_fp32_ms, ms) for name, (ms, _) in variants.items()}

    quality_vs_fp32 = {
        name: score_error_metrics(scores_ref=dense_fp32_scores, scores_test=out)
        for name, (_, out) in variants.items()
    }

    result = {
        "T": int(T),
        "shape": {"B": int(B), "H": int(H), "Q": 1, "D": int(D), "M": int(M)},
        "timing_ms": timing_ms,
        "speedup_over_dense_fp16_einsum": speedup_over_fp16,
        "speedup_over_dense_fp32_qkt": speedup_over_fp32,
        "quality_vs_dense_fp32_qkt": quality_vs_fp32,
        "memory_bytes": {
            "dense_fp16_k_bytes": int(dense_fp16_k_bytes),
            "dense_fp32_k_bytes": int(dense_fp32_k_bytes),
            "compressed_prod_k_bytes": int(compressed_prod_k_bytes),
            "mse_1bit_total_bytes": int(mse_1bit_total_bytes),
            "mse_4bit_total_bytes": int(mse_4bit_total_bytes),
        },
        "memory_ratio": {
            "prod_k_over_dense_fp16_k": float(compressed_prod_k_bytes / dense_fp16_k_bytes),
            "prod_k_over_dense_fp32_k": float(compressed_prod_k_bytes / dense_fp32_k_bytes),
            "mse_1bit_over_dense_fp16_k": float(mse_1bit_total_bytes / dense_fp16_k_bytes),
            "mse_1bit_over_dense_fp32_k": float(mse_1bit_total_bytes / dense_fp32_k_bytes),
            "mse_4bit_over_dense_fp16_k": float(mse_4bit_total_bytes / dense_fp16_k_bytes),
            "mse_4bit_over_dense_fp32_k": float(mse_4bit_total_bytes / dense_fp32_k_bytes),
        },
        "prod_cache_report": layer_report,
    }

    print(f"dense FP16 einsum: {dense_fp16_ms:.6f} ms")
    print(f"dense FP32 qK^T:   {dense_fp32_ms:.6f} ms")
    for name, (ms, _) in variants.items():
        print(
            f"{name:36s} {ms:.6f} ms | "
            f"vs FP16 einsum {speedup_over_fp16[name]:.4f}x | "
            f"vs FP32 qK^T {speedup_over_fp32[name]:.4f}x"
        )

    del cache, dense_k_fp16, dense_k_fp32, q_fp16, q_fp32
    del rotation, centroids_2bit, sketch
    del packed_mse_2bit, packed_qjl, packed_mse_2bit_t, packed_qjl_t, mse_norms_2bit, qjl_norms
    del packed_1bit_t, mse_norms_1bit, centroids_1bit
    del packed_4bit_t, mse_norms_4bit, centroids_4bit
    del dense_fp16_scores, dense_fp32_scores
    for _, out in variants.values():
        del out
    torch.cuda.empty_cache()

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rotation-based TurboQuant paper-base speedup against two dense baselines."
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--qjl_m", type=int, default=256)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seq_lens", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    p.add_argument("--build_chunk_size", type=int, default=1024)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = ensure_cuda(args.device)

    B = int(args.batch_size)
    H = int(args.num_heads)
    D = int(args.head_dim)
    M = int(args.qjl_m)

    print("========== Rotation-based TurboQuant dual-baseline benchmark ==========")
    print(f"device           = {device}")
    print(f"B                = {B}")
    print(f"H                = {H}")
    print(f"Q                = 1")
    print(f"D                = {D}")
    print(f"M                = {M}")
    print(f"warmup           = {args.warmup}")
    print(f"iters            = {args.iters}")
    print(f"seq_lens         = {list(args.seq_lens)}")
    print(f"build_chunk_size = {args.build_chunk_size}")
    print("[Baseline A] dense FP16 einsum")
    print("[Baseline B] dense FP32 qK^T")

    results = [
        benchmark_one_length(
            T=int(T),
            B=B,
            H=H,
            D=D,
            M=M,
            device=device,
            warmup=int(args.warmup),
            iters=int(args.iters),
            build_chunk_size=int(args.build_chunk_size),
        )
        for T in args.seq_lens
    ]

    payload = {
        "benchmark": "rotation_turboquant_dual_baseline_speedup",
        "method": "rotation_turboquant_paper_base",
        "device": str(device),
        "config": {
            "B": B,
            "H": H,
            "Q": 1,
            "D": D,
            "M": M,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "seq_lens": [int(x) for x in args.seq_lens],
            "build_chunk_size": int(args.build_chunk_size),
        },
        "baselines": {
            "dense_fp16_einsum": "torch.einsum('bhqd,bhkd->bhqk', q_fp16, k_fp16)",
            "dense_fp32_qkt": "torch.matmul(q_fp32, k_fp32.transpose(-1,-2))",
        },
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"[Save] {out_path}")
    print("[PASS] Rotation-based TurboQuant dual-baseline benchmark completed.")


if __name__ == "__main__":
    main()
