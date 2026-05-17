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

from turboquant.polarquant import recursive_polar_encode
from turboquant.polarquant_quant import (
    DEFAULT_POLAR_BITS_BY_LEVEL,
    fit_polar_angle_codebooks_from_encodings,
)
from turboquant.polar_prod import turboquant_polar_prod_quantize
from turboquant.qjl import make_gaussian_sketch
from turboquant.turboquant_logits_cuda import turboquant_fused_logits_cuda
from turboquant.packed_meta import build_turboquant_packed_meta_blob


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
def build_codebooks_and_sketch(
    *,
    device: torch.device,
    d: int,
    m: int,
    num_levels: int,
    n_calib: int,
    seed: int,
) -> tuple[Any, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x_calib = torch.randn(
        n_calib,
        d,
        device=device,
        dtype=torch.float32,
    )

    enc_calib = recursive_polar_encode(
        x_calib,
        num_levels=num_levels,
    )

    codebooks = fit_polar_angle_codebooks_from_encodings(
        [enc_calib],
        bits_by_level=DEFAULT_POLAR_BITS_BY_LEVEL,
        max_iters=30,
        max_samples_per_level=200_000,
        seed=seed,
    )

    sketch = make_gaussian_sketch(
        d=d,
        m=m,
        device=device,
        dtype=torch.float32,
        seed=seed + 123,
    )

    del x_calib
    del enc_calib
    torch.cuda.empty_cache()
    return codebooks, sketch


@torch.no_grad()
def benchmark_one_length(
    *,
    seq_len: int,
    device: torch.device,
    B: int,
    H: int,
    Q: int,
    D: int,
    M: int,
    num_levels: int,
    warmup: int,
    iters: int,
    seed: int,
    codebooks: Any,
    sketch: torch.Tensor,
) -> dict[str, Any]:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")

    torch.manual_seed(seed + int(seq_len))
    torch.cuda.manual_seed_all(seed + int(seq_len))

    # ------------------------------------------------------------
    # Dense tensors for the two baselines
    # ------------------------------------------------------------
    q_fp32 = torch.randn(B, H, Q, D, device=device, dtype=torch.float32)
    k_fp32 = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float32)
    q_fp16 = q_fp32.to(torch.float16)
    k_fp16 = k_fp32.to(torch.float16)

    # ------------------------------------------------------------
    # PolarQuant-based compressed K representation
    # ------------------------------------------------------------
    encoding = turboquant_polar_prod_quantize(
        x=k_fp32,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=num_levels,
    )

    packed = encoding.polar.packed_angles
    packed_qjl_signs = encoding.qjl_residual.packed_sign_bits.reshape(
        B,
        H,
        seq_len,
        M // 8,
    ).contiguous()
    qjl_norms = encoding.qjl_residual.norms.reshape(B, H, seq_len).contiguous()

    packed_meta = build_turboquant_packed_meta_blob(
        packed_l1=packed.level1_4bit,
        packed_l2=packed.level2_2bit,
        packed_l3=packed.level3_2bit,
        packed_l4=packed.level4_2bit,
        packed_qjl_signs=packed_qjl_signs,
    )

    q_projected = torch.matmul(
        q_fp32,
        sketch.T.to(torch.float32),
    ).contiguous()

    # ------------------------------------------------------------
    # Baselines and compressed score path
    # ------------------------------------------------------------
    def dense_fp16_einsum_logits() -> torch.Tensor:
        return torch.einsum("bhqd,bhkd->bhqk", q_fp16, k_fp16)

    def dense_fp32_qkt_logits() -> torch.Tensor:
        return torch.matmul(q_fp32, k_fp32.transpose(-1, -2))

    def polarquant_logits() -> torch.Tensor:
        return turboquant_fused_logits_cuda(
            q=q_fp32,
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
            packed_meta=packed_meta,
        )

    dense_fp16_einsum_ms, dense_fp16_scores = bench_cuda_ms(
        dense_fp16_einsum_logits,
        warmup=warmup,
        iters=iters,
    )
    dense_fp32_qkt_ms, dense_fp32_scores = bench_cuda_ms(
        dense_fp32_qkt_logits,
        warmup=warmup,
        iters=iters,
    )
    polarquant_ms, polarquant_scores = bench_cuda_ms(
        polarquant_logits,
        warmup=warmup,
        iters=iters,
    )

    quality_vs_fp32 = score_error_metrics(
        scores_ref=dense_fp32_scores,
        scores_test=polarquant_scores,
    )

    dense_fp16_k_bytes = tensor_bytes(k_fp16)
    dense_fp32_k_bytes = tensor_bytes(k_fp32)

    logical_tq_k_bytes = (
        tensor_bytes(packed.level1_4bit)
        + tensor_bytes(packed.level2_2bit)
        + tensor_bytes(packed.level3_2bit)
        + tensor_bytes(packed.level4_2bit)
        + tensor_bytes(packed_qjl_signs)
        + tensor_bytes(encoding.polar.radii)
        + tensor_bytes(qjl_norms)
    )

    meta64_physical_tq_k_bytes = (
        tensor_bytes(packed_meta)
        + tensor_bytes(encoding.polar.radii)
        + tensor_bytes(qjl_norms)
    )

    num_k_channels = B * H * seq_len * D
    logical_k_bpc = float(logical_tq_k_bytes) * 8.0 / float(num_k_channels)
    meta64_k_bpc = float(meta64_physical_tq_k_bytes) * 8.0 / float(num_k_channels)

    result = {
        "seq_len": int(seq_len),
        "shape": {"B": int(B), "H": int(H), "Q": int(Q), "D": int(D), "M": int(M)},
        "timing_ms": {
            "dense_fp16_einsum_ms": float(dense_fp16_einsum_ms),
            "dense_fp32_qkt_ms": float(dense_fp32_qkt_ms),
            "polarquant_fused_logits_ms": float(polarquant_ms),
        },
        "speedup": {
            "over_dense_fp16_einsum": safe_ratio(dense_fp16_einsum_ms, polarquant_ms),
            "over_dense_fp32_qkt": safe_ratio(dense_fp32_qkt_ms, polarquant_ms),
        },
        "quality_vs_dense_fp32_qkt": quality_vs_fp32,
        "memory_bytes": {
            "dense_fp16_k_bytes": int(dense_fp16_k_bytes),
            "dense_fp32_k_bytes": int(dense_fp32_k_bytes),
            "polarquant_logical_k_bytes": int(logical_tq_k_bytes),
            "polarquant_meta64_physical_k_bytes": int(meta64_physical_tq_k_bytes),
        },
        "memory_ratio": {
            "logical_over_dense_fp16_k": float(logical_tq_k_bytes / dense_fp16_k_bytes),
            "logical_over_dense_fp32_k": float(logical_tq_k_bytes / dense_fp32_k_bytes),
            "meta64_physical_over_dense_fp16_k": float(meta64_physical_tq_k_bytes / dense_fp16_k_bytes),
            "meta64_physical_over_dense_fp32_k": float(meta64_physical_tq_k_bytes / dense_fp32_k_bytes),
        },
        "effective_k_bits_per_channel": {
            "logical": float(logical_k_bpc),
            "meta64_physical": float(meta64_k_bpc),
        },
    }

    print()
    print("=" * 72)
    print(f"[PolarQuant dual-baseline benchmark] T={seq_len}")
    print("=" * 72)
    print(f"dense FP16 einsum:      {dense_fp16_einsum_ms:.6f} ms")
    print(f"dense FP32 qK^T:        {dense_fp32_qkt_ms:.6f} ms")
    print(f"PolarQuant fused logits:{polarquant_ms:.6f} ms")
    print(f"speedup vs FP16 einsum: {result['speedup']['over_dense_fp16_einsum']:.4f}x")
    print(f"speedup vs FP32 qK^T:   {result['speedup']['over_dense_fp32_qkt']:.4f}x")
    print(f"meta64 physical K bpc:  {meta64_k_bpc:.6f}")
    print(f"relative RMSE vs FP32:  {quality_vs_fp32['relative_rmse']:.6e}")

    del q_fp32, q_fp16, k_fp32, k_fp16
    del encoding, packed, packed_qjl_signs, qjl_norms, packed_meta, q_projected
    del dense_fp16_scores, dense_fp32_scores, polarquant_scores
    torch.cuda.empty_cache()

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PolarQuant-based compressed logits speedup against two dense baselines."
    )
    p.add_argument("--seq_lens", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--num_queries", type=int, default=1)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--qjl_dim", type=int, default=128)
    p.add_argument("--num_levels", type=int, default=4)
    p.add_argument("--n_calib", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = ensure_cuda(args.device)

    B = int(args.batch_size)
    H = int(args.num_heads)
    Q = int(args.num_queries)
    D = int(args.head_dim)
    M = int(args.qjl_dim)
    num_levels = int(args.num_levels)

    if D != 128:
        raise ValueError(f"Current fused PolarQuant logits kernel expects D=128, got D={D}.")
    if M != 128:
        raise ValueError(f"Current fused PolarQuant logits kernel expects M=128, got M={M}.")

    print("========== PolarQuant dual-baseline attention-logits benchmark ==========")
    print(f"device       = {device}")
    print(f"B            = {B}")
    print(f"H            = {H}")
    print(f"Q            = {Q}")
    print(f"D            = {D}")
    print(f"M            = {M}")
    print(f"num_levels   = {num_levels}")
    print(f"warmup       = {args.warmup}")
    print(f"iters        = {args.iters}")
    print(f"seq_lens     = {list(args.seq_lens)}")
    print("[Baseline A] dense FP16 einsum")
    print("[Baseline B] dense FP32 qK^T")

    codebooks, sketch = build_codebooks_and_sketch(
        device=device,
        d=D,
        m=M,
        num_levels=num_levels,
        n_calib=int(args.n_calib),
        seed=int(args.seed),
    )

    results = [
        benchmark_one_length(
            seq_len=int(seq_len),
            device=device,
            B=B,
            H=H,
            Q=Q,
            D=D,
            M=M,
            num_levels=num_levels,
            warmup=int(args.warmup),
            iters=int(args.iters),
            seed=int(args.seed),
            codebooks=codebooks,
            sketch=sketch,
        )
        for seq_len in args.seq_lens
    ]

    payload = {
        "benchmark": "polarquant_dual_baseline_speedup",
        "method": "polarquant_base",
        "device": str(device),
        "config": {
            "B": B,
            "H": H,
            "Q": Q,
            "D": D,
            "M": M,
            "num_levels": num_levels,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "seq_lens": [int(x) for x in args.seq_lens],
            "seed": int(args.seed),
            "n_calib": int(args.n_calib),
            "bits_by_level": list(DEFAULT_POLAR_BITS_BY_LEVEL),
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
    print("[PASS] PolarQuant dual-baseline benchmark completed.")


if __name__ == "__main__":
    main()
