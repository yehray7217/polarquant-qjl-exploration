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
from turboquant.packed_meta import build_turboquant_packed_meta_blob
from turboquant.polar_tree_lut import build_tree_l2_factor_lut
from turboquant.qjl_sign_layout import build_qjl_lane_nibble_signs
from turboquant.turboquant_l2_combined_factor_lut_lane_nibble_cuda import (
    turboquant_polar_tree_l2_combined_lut_polar_only_cuda,
    turboquant_polar_tree_l2_combined_lut_lane_nibble_fused_logits_cuda,
)
from turboquant.turboquant_l2_combined_factor_lut_fp16_storage_cuda import (
    turboquant_polar_tree_l2_combined_lut_fp16_polar_only_cuda,
    turboquant_polar_tree_l2_combined_lut_fp16_lane_nibble_fused_logits_cuda,
)


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
    return float(start.elapsed_time(end) / iters), out


@torch.no_grad()
def parity_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    aa = a.to(torch.float32)
    bb = b.to(torch.float32)
    diff = (aa - bb).abs()
    ref_abs_mean = float(aa.abs().mean().item())
    ref_rms = float(torch.sqrt(torch.mean(aa * aa)).item())
    mae = float(diff.mean().item())
    rmse = float(torch.sqrt(torch.mean((aa - bb) * (aa - bb))).item())
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": mae,
        "rmse": rmse,
        "relative_mae": float(mae / ref_abs_mean) if ref_abs_mean != 0.0 else float("inf"),
        "relative_rmse": float(rmse / ref_rms) if ref_rms != 0.0 else float("inf"),
    }


def tensor_bytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())


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
    torch.manual_seed(seed + int(seq_len))
    torch.cuda.manual_seed_all(seed + int(seq_len))

    q_fp32 = torch.randn(B, H, Q, D, device=device, dtype=torch.float32)
    k_fp32 = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float32)

    encoding = turboquant_polar_prod_quantize(
        x=k_fp32,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=num_levels,
    )

    packed = encoding.polar.packed_angles
    packed_qjl_signs = encoding.qjl_residual.packed_sign_bits.reshape(
        B, H, seq_len, M // 8
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

    l2_factor_lut_fp32 = build_tree_l2_factor_lut(
        q=q_fp32,
        centroids_l1=codebooks.centroids[0],
        centroids_l2=codebooks.centroids[1],
    )
    l2_factor_lut_fp16 = l2_factor_lut_fp32.to(torch.float16).contiguous()

    lane_nibble_qjl_signs = build_qjl_lane_nibble_signs(
        packed_qjl_signs
    )

    def polar_fp32() -> torch.Tensor:
        return turboquant_polar_tree_l2_combined_lut_polar_only_cuda(
            packed_meta=packed_meta,
            radii=encoding.polar.radii,
            l2_factor_lut=l2_factor_lut_fp32,
            centroids_l3=codebooks.centroids[2],
            centroids_l4=codebooks.centroids[3],
        )

    def polar_fp16_storage() -> torch.Tensor:
        return turboquant_polar_tree_l2_combined_lut_fp16_polar_only_cuda(
            packed_meta=packed_meta,
            radii=encoding.polar.radii,
            l2_factor_lut_fp16=l2_factor_lut_fp16,
            centroids_l3=codebooks.centroids[2],
            centroids_l4=codebooks.centroids[3],
        )

    def full_fp32() -> torch.Tensor:
        return turboquant_polar_tree_l2_combined_lut_lane_nibble_fused_logits_cuda(
            q_projected=q_projected,
            packed_meta=packed_meta,
            radii=encoding.polar.radii,
            l2_factor_lut=l2_factor_lut_fp32,
            centroids_l3=codebooks.centroids[2],
            centroids_l4=codebooks.centroids[3],
            lane_nibble_qjl_signs=lane_nibble_qjl_signs,
            qjl_norms=qjl_norms,
        )

    def full_fp16_storage() -> torch.Tensor:
        return turboquant_polar_tree_l2_combined_lut_fp16_lane_nibble_fused_logits_cuda(
            q_projected=q_projected,
            packed_meta=packed_meta,
            radii=encoding.polar.radii,
            l2_factor_lut_fp16=l2_factor_lut_fp16,
            centroids_l3=codebooks.centroids[2],
            centroids_l4=codebooks.centroids[3],
            lane_nibble_qjl_signs=lane_nibble_qjl_signs,
            qjl_norms=qjl_norms,
        )

    polar_fp32_ms, polar_fp32_scores = bench_cuda_ms(
        polar_fp32,
        warmup=warmup,
        iters=iters,
    )
    polar_fp16_ms, polar_fp16_scores = bench_cuda_ms(
        polar_fp16_storage,
        warmup=warmup,
        iters=iters,
    )
    full_fp32_ms, full_fp32_scores = bench_cuda_ms(
        full_fp32,
        warmup=warmup,
        iters=iters,
    )
    full_fp16_ms, full_fp16_scores = bench_cuda_ms(
        full_fp16_storage,
        warmup=warmup,
        iters=iters,
    )

    result = {
        "seq_len": int(seq_len),
        "shape": {"B": B, "H": H, "Q": Q, "D": D, "M": M},
        "timing_ms": {
            "polar_only_l2_lut_fp32_storage_ms": float(polar_fp32_ms),
            "polar_only_l2_lut_fp16_storage_ms": float(polar_fp16_ms),
            "full_lane_nibble_l2_lut_fp32_storage_ms": float(full_fp32_ms),
            "full_lane_nibble_l2_lut_fp16_storage_ms": float(full_fp16_ms),
        },
        "speedup": {
            "polar_fp16_storage_over_fp32_storage": float(polar_fp32_ms / polar_fp16_ms),
            "full_fp16_storage_over_fp32_storage": float(full_fp32_ms / full_fp16_ms),
        },
        "difference_vs_fp32_storage": {
            "polar_fp16_vs_fp32_storage": parity_metrics(
                polar_fp16_scores,
                polar_fp32_scores,
            ),
            "full_fp16_vs_fp32_storage": parity_metrics(
                full_fp16_scores,
                full_fp32_scores,
            ),
        },
        "factor_lut_bytes": {
            "l2_factor_lut_fp32_storage_bytes": int(tensor_bytes(l2_factor_lut_fp32)),
            "l2_factor_lut_fp16_storage_bytes": int(tensor_bytes(l2_factor_lut_fp16)),
        },
    }

    print(json.dumps(result, indent=2))

    del q_fp32, k_fp32
    del encoding, packed, packed_qjl_signs, qjl_norms, packed_meta
    del q_projected, l2_factor_lut_fp32, l2_factor_lut_fp16, lane_nibble_qjl_signs
    del polar_fp32_scores, polar_fp16_scores, full_fp32_scores, full_fp16_scores
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "FP16 storage benchmark for L2 combined factor LUT: "
            "FP32 storage vs FP16 storage, on Polar-only and "
            "full lane-nibble fused score paths."
        )
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

    if B != 1 or Q != 1:
        raise ValueError("Current FP16 L2 LUT fast paths require B=1 and Q=1.")
    if D != 128 or M != 128 or num_levels != 4:
        raise ValueError("Current FP16 L2 LUT fast paths require D=128, M=128, num_levels=4.")

    print("========== L2 combined factor LUT FP16 storage benchmark ==========")
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
        "[Note] FP16 storage keeps the same L2 combined LUT index and "
        "converts each half lookup value back to FP32 before downstream "
        "Polar tree arithmetic."
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
        "benchmark": "polarquant_l2_combined_factor_lut_fp16_storage",
        "method": "l2_combined_factor_lut_fp32_storage_vs_fp16_storage_lane_nibble",
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
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"[Save] {out_path}")
    print("[PASS] L2 combined factor LUT FP16 storage benchmark completed.")


if __name__ == "__main__":
    main()
