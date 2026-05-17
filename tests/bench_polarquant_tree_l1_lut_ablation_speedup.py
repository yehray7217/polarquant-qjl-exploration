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
from turboquant.polar_tree_lut import build_tree_l1_factor_lut
from turboquant.turboquant_logits_tree_lut_cuda import (
    turboquant_polar_tree_l1_lut_fused_logits_cuda,
)
from turboquant.turboquant_logits_tree_l1_lut_ablation_cuda import (
    turboquant_polar_tree_l1_lut_polar_only_cuda,
    turboquant_polar_tree_l1_lut_qjl_only_cuda,
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

    l1_factor_lut = build_tree_l1_factor_lut(
        q=q_fp32,
        centroids_l1=codebooks.centroids[0],
    )

    def full_l1_lut() -> torch.Tensor:
        return turboquant_polar_tree_l1_lut_fused_logits_cuda(
            q_projected=q_projected,
            packed_meta=packed_meta,
            radii=encoding.polar.radii,
            l1_factor_lut=l1_factor_lut,
            centroids_l2=codebooks.centroids[1],
            centroids_l3=codebooks.centroids[2],
            centroids_l4=codebooks.centroids[3],
            qjl_norms=qjl_norms,
        )

    def polar_only() -> torch.Tensor:
        return turboquant_polar_tree_l1_lut_polar_only_cuda(
            packed_meta=packed_meta,
            radii=encoding.polar.radii,
            l1_factor_lut=l1_factor_lut,
            centroids_l2=codebooks.centroids[1],
            centroids_l3=codebooks.centroids[2],
            centroids_l4=codebooks.centroids[3],
        )

    def qjl_only() -> torch.Tensor:
        return turboquant_polar_tree_l1_lut_qjl_only_cuda(
            q_projected=q_projected,
            packed_meta=packed_meta,
            qjl_norms=qjl_norms,
        )

    full_ms, full_scores = bench_cuda_ms(
        full_l1_lut,
        warmup=warmup,
        iters=iters,
    )
    polar_ms, polar_scores = bench_cuda_ms(
        polar_only,
        warmup=warmup,
        iters=iters,
    )
    qjl_ms, qjl_scores = bench_cuda_ms(
        qjl_only,
        warmup=warmup,
        iters=iters,
    )

    reconstructed_full = polar_scores + qjl_scores

    result = {
        "seq_len": int(seq_len),
        "shape": {"B": B, "H": H, "Q": Q, "D": D, "M": M},
        "timing_ms": {
            "full_l1_lut_ms": float(full_ms),
            "polar_only_l1_lut_ms": float(polar_ms),
            "qjl_only_ms": float(qjl_ms),
            "polar_plus_qjl_separate_kernel_sum_ms": float(polar_ms + qjl_ms),
        },
        "relative_to_full": {
            "polar_only_over_full": float(polar_ms / full_ms),
            "qjl_only_over_full": float(qjl_ms / full_ms),
            "separate_sum_over_full": float((polar_ms + qjl_ms) / full_ms),
        },
        "parity_full_vs_polar_plus_qjl": parity_metrics(
            full_scores,
            reconstructed_full,
        ),
    }

    print(json.dumps(result, indent=2))

    del q_fp32, k_fp32
    del encoding, packed, packed_qjl_signs, qjl_norms, packed_meta, q_projected, l1_factor_lut
    del full_scores, polar_scores, qjl_scores, reconstructed_full
    torch.cuda.empty_cache()

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "PolarQuant L1-LUT score ablation: "
            "full fused vs Polar-only vs QJL-only."
        )
    )
    p.add_argument("--seq_lens", type=int, nargs="+", default=[65536])
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
        raise ValueError("Current L1-LUT ablation fast paths require B=1 and Q=1.")
    if D != 128 or M != 128 or num_levels != 4:
        raise ValueError("Current L1-LUT ablation fast paths require D=128, M=128, num_levels=4.")

    print("========== PolarQuant L1-LUT score ablation benchmark ==========")
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
        "[Note] Keeps the 5.125-bpc L1-LUT score path fixed and decomposes "
        "the score into Polar-only and QJL-only CUDA kernels. "
        "The JSON checks Full ≈ Polar-only + QJL-only."
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
        "benchmark": "polarquant_tree_l1_lut_ablation_speedup",
        "method": "polarquant_meta64_m128_bits_4222_tree_l1_lut_ablation",
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
    print("[PASS] PolarQuant L1-LUT score ablation benchmark completed.")


if __name__ == "__main__":
    main()
