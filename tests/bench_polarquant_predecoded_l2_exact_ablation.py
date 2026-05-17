#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).parent.name == "tests"
    else Path.cwd()
)
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
from turboquant.predecoded_l2_layout import build_predecoded_l2_codes_exact
from turboquant.turboquant_qjl_norm_early_load_cuda import (
    turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda,
)
from turboquant.turboquant_predecoded_l2_exact_ablation_cuda import (
    turboquant_predecoded_l2_exact_fused_logits_cuda,
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

    del x_calib, enc_calib
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
        B,
        H,
        seq_len,
        M // 8,
    ).contiguous()
    qjl_norms = encoding.qjl_residual.norms.reshape(
        B,
        H,
        seq_len,
    ).contiguous()

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

    l2_lut_fp32 = build_tree_l2_factor_lut(
        q=q_fp32,
        centroids_l1=codebooks.centroids[0],
        centroids_l2=codebooks.centroids[1],
    )
    l2_lut_fp16 = l2_lut_fp32.to(torch.float16).contiguous()
    lane_nibble_qjl_signs = build_qjl_lane_nibble_signs(
        packed_qjl_signs
    )
    predecoded_l2_codes = build_predecoded_l2_codes_exact(
        packed.level2_2bit
    )

    common = dict(
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=encoding.polar.radii,
        l2_factor_lut_fp16=l2_lut_fp16,
        centroids_l3=codebooks.centroids[2],
        centroids_l4=codebooks.centroids[3],
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        qjl_norms=qjl_norms,
    )

    def baseline() -> torch.Tensor:
        return turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda(
            **common
        )

    def predecoded_l2_exact() -> torch.Tensor:
        return turboquant_predecoded_l2_exact_fused_logits_cuda(
            **common,
            predecoded_l2_codes=predecoded_l2_codes,
        )

    baseline_ms, baseline_out = bench_cuda_ms(
        baseline,
        warmup=warmup,
        iters=iters,
    )
    candidate_ms, candidate_out = bench_cuda_ms(
        predecoded_l2_exact,
        warmup=warmup,
        iters=iters,
    )

    result = {
        "seq_len": int(seq_len),
        "shape": {"B": B, "H": H, "Q": Q, "D": D, "M": M},
        "timing_ms": {
            "baseline_current_best_ms": float(baseline_ms),
            "predecoded_l2_exact_ms": float(candidate_ms),
        },
        "speedup_vs_baseline": {
            "predecoded_l2_exact_over_baseline": float(
                baseline_ms / candidate_ms
            ),
        },
        "parity_vs_baseline": {
            "predecoded_l2_exact": parity_metrics(
                candidate_out,
                baseline_out,
            ),
        },
        "storage_bytes": {
            "packed_l2_2bit_bytes": int(tensor_bytes(packed.level2_2bit)),
            "predecoded_l2_uint8_bytes": int(tensor_bytes(predecoded_l2_codes)),
            "predecoded_l2_extra_bytes_vs_packed_l2": int(
                tensor_bytes(predecoded_l2_codes)
                - tensor_bytes(packed.level2_2bit)
            ),
        },
        "ablation_note": (
            "The predecoded L2 tensor contains exact original c2 values. "
            "The L2 combined LUT index is therefore identical to baseline; "
            "the comparison isolates the packed L2 decode path versus a "
            "direct uint8 c2 load, modulo the candidate tensor's larger memory footprint."
        ),
    }

    print(json.dumps(result, indent=2))

    del q_fp32, k_fp32
    del encoding, packed, packed_qjl_signs, qjl_norms, packed_meta
    del q_projected, l2_lut_fp32, l2_lut_fp16, lane_nibble_qjl_signs
    del predecoded_l2_codes
    del baseline_out, candidate_out
    torch.cuda.empty_cache()

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Exact-value predecoded L2 code ablation for the current "
            "L2-combined-FP16-LUT + early-radii + early-qjl-norm full score path."
        )
    )
    p.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[16384, 32768, 65536, 131072],
    )
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
        raise ValueError("Current predecoded-L2 fast path requires B=1 and Q=1.")
    if D != 128 or M != 128 or num_levels != 4:
        raise ValueError(
            "Current predecoded-L2 fast path requires "
            "D=128, M=128, num_levels=4."
        )

    print("========== Exact-value predecoded L2 code ablation ==========")
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
        "[Note] Baseline is FP16 L2 LUT + lane-nibble QJL + early-radii "
        "+ early-qjl-norm. Candidate reads exact predecoded uint8 L2 codes, "
        "keeping the combined LUT index identical to baseline."
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
        "benchmark": "polarquant_predecoded_l2_exact_ablation",
        "method": "current_best_vs_exact_value_predecoded_l2_uint8",
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
        },
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"[Save] {out_path}")
    print("[PASS] Exact-value predecoded L2 code ablation completed.")


if __name__ == "__main__":
    main()
