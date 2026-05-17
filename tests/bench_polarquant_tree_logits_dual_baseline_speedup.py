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
from turboquant.turboquant_logits_tree_cuda import (
    turboquant_polar_tree_fused_logits_cuda,
)
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
def parity_metrics(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, float]:
    aa = a.to(torch.float32)
    bb = b.to(torch.float32)
    diff = (aa - bb).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
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

    q_fp32 = torch.randn(B, H, Q, D, device=device, dtype=torch.float32)
    k_fp32 = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float32)
    q_fp16 = q_fp32.to(torch.float16)
    k_fp16 = k_fp32.to(torch.float16)

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

    def dense_fp16_einsum_logits() -> torch.Tensor:
        return torch.einsum("bhqd,bhkd->bhqk", q_fp16, k_fp16)

    def dense_fp32_qkt_logits() -> torch.Tensor:
        return torch.matmul(q_fp32, k_fp32.transpose(-1, -2))

    def polarquant_coordinate_logits() -> torch.Tensor:
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

    def polarquant_tree_logits() -> torch.Tensor:
        return turboquant_polar_tree_fused_logits_cuda(
            q=q_fp32,
            q_projected=q_projected,
            packed_meta=packed_meta,
            radii=encoding.polar.radii,
            centroids_l1=codebooks.centroids[0],
            centroids_l2=codebooks.centroids[1],
            centroids_l3=codebooks.centroids[2],
            centroids_l4=codebooks.centroids[3],
            qjl_norms=qjl_norms,
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
    coordinate_ms, coordinate_scores = bench_cuda_ms(
        polarquant_coordinate_logits,
        warmup=warmup,
        iters=iters,
    )
    tree_ms, tree_scores = bench_cuda_ms(
        polarquant_tree_logits,
        warmup=warmup,
        iters=iters,
    )

    coordinate_quality = score_error_metrics(
        scores_ref=dense_fp32_scores,
        scores_test=coordinate_scores,
    )
    tree_quality = score_error_metrics(
        scores_ref=dense_fp32_scores,
        scores_test=tree_scores,
    )
    tree_vs_coordinate = parity_metrics(
        a=tree_scores,
        b=coordinate_scores,
    )

    dense_fp16_k_bytes = tensor_bytes(k_fp16)
    dense_fp32_k_bytes = tensor_bytes(k_fp32)

    logical_k_bytes = (
        tensor_bytes(packed.level1_4bit)
        + tensor_bytes(packed.level2_2bit)
        + tensor_bytes(packed.level3_2bit)
        + tensor_bytes(packed.level4_2bit)
        + tensor_bytes(packed_qjl_signs)
        + tensor_bytes(encoding.polar.radii)
        + tensor_bytes(qjl_norms)
    )

    physical_meta64_k_bytes = (
        tensor_bytes(packed_meta)
        + tensor_bytes(encoding.polar.radii)
        + tensor_bytes(qjl_norms)
    )

    num_k_channels = B * H * seq_len * D
    logical_bpc = float(logical_k_bytes) * 8.0 / float(num_k_channels)
    physical_bpc = float(physical_meta64_k_bytes) * 8.0 / float(num_k_channels)

    result = {
        "seq_len": int(seq_len),
        "shape": {
            "B": B,
            "H": H,
            "Q": Q,
            "D": D,
            "M": M,
        },
        "timing_ms": {
            "dense_fp16_einsum_ms": float(dense_fp16_einsum_ms),
            "dense_fp32_qkt_ms": float(dense_fp32_qkt_ms),
            "polarquant_coordinate_fused_logits_ms": float(coordinate_ms),
            "polarquant_tree_fused_logits_ms": float(tree_ms),
        },
        "speedup": {
            "tree_over_dense_fp16_einsum": float(dense_fp16_einsum_ms / tree_ms),
            "tree_over_dense_fp32_qkt": float(dense_fp32_qkt_ms / tree_ms),
            "tree_over_coordinate_fused_logits": float(coordinate_ms / tree_ms),
            "coordinate_over_dense_fp16_einsum": float(dense_fp16_einsum_ms / coordinate_ms),
            "coordinate_over_dense_fp32_qkt": float(dense_fp32_qkt_ms / coordinate_ms),
        },
        "quality_vs_dense_fp32_qkt": {
            "coordinate_fused_logits": coordinate_quality,
            "tree_fused_logits": tree_quality,
        },
        "parity_tree_vs_coordinate": tree_vs_coordinate,
        "memory_bytes": {
            "dense_fp16_k_bytes": int(dense_fp16_k_bytes),
            "dense_fp32_k_bytes": int(dense_fp32_k_bytes),
            "polarquant_logical_k_bytes": int(logical_k_bytes),
            "polarquant_meta64_physical_k_bytes": int(physical_meta64_k_bytes),
        },
        "effective_k_bits_per_channel": {
            "logical": float(logical_bpc),
            "meta64_physical": float(physical_bpc),
        },
    }

    print(json.dumps(result, indent=2))

    del q_fp32
    del k_fp32
    del q_fp16
    del k_fp16
    del encoding
    del packed
    del packed_qjl_signs
    del qjl_norms
    del packed_meta
    del q_projected
    del dense_fp16_scores
    del dense_fp32_scores
    del coordinate_scores
    del tree_scores
    torch.cuda.empty_cache()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PolarQuant direct-tree logits benchmark: "
            "existing coordinate-style fused logits vs direct Polar tree fused logits."
        )
    )
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_queries", type=int, default=1)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--qjl_dim", type=int, default=128)
    parser.add_argument("--num_levels", type=int, default=4)
    parser.add_argument("--n_calib", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = ensure_cuda(args.device)

    B = int(args.batch_size)
    H = int(args.num_heads)
    Q = int(args.num_queries)
    D = int(args.head_dim)
    M = int(args.qjl_dim)
    num_levels = int(args.num_levels)

    if B != 1 or Q != 1:
        raise ValueError("Current Polar tree CUDA fast path requires B=1 and Q=1.")
    if D != 128:
        raise ValueError(f"Current Polar tree CUDA fast path requires D=128, got D={D}.")
    if M != 128:
        raise ValueError(f"Current Polar tree CUDA fast path requires M=128, got M={M}.")
    if num_levels != 4:
        raise ValueError(f"Current Polar tree CUDA fast path requires num_levels=4, got {num_levels}.")

    print("========== PolarQuant direct-tree logits benchmark ==========")
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
    print()
    print(
        "[Note] This keeps the existing 5.125-physical-bpc PolarQuant format "
        "and compares the current coordinate-style fused logits against a "
        "direct recursive Polar-tree dot-product formulation."
    )

    codebooks, sketch = build_codebooks_and_sketch(
        device=device,
        d=D,
        m=M,
        num_levels=num_levels,
        n_calib=int(args.n_calib),
        seed=int(args.seed),
    )

    results = []
    for seq_len in args.seq_lens:
        print()
        print("=" * 78)
        print(f"[Benchmark] T={int(seq_len)}")
        print("=" * 78)
        results.append(
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
        )

    payload = {
        "benchmark": "polarquant_tree_logits_dual_baseline_speedup",
        "method": "polarquant_meta64_m128_bits_4222_direct_tree",
        "device": str(device),
        "config": {
            "B": B,
            "H": H,
            "Q": Q,
            "D": D,
            "M": M,
            "num_levels": num_levels,
            "bits_by_level": list(DEFAULT_POLAR_BITS_BY_LEVEL),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "seq_lens": [int(x) for x in args.seq_lens],
            "seed": int(args.seed),
            "n_calib": int(args.n_calib),
        },
        "baselines": {
            "dense_fp16_einsum": "torch.einsum('bhqd,bhkd->bhqk', q_fp16, k_fp16)",
            "dense_fp32_qkt": "torch.matmul(q_fp32, k_fp32.transpose(-1,-2))",
        },
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"[Save] {out_path}")
    print("[PASS] PolarQuant direct-tree logits benchmark completed.")


if __name__ == "__main__":
    main()
