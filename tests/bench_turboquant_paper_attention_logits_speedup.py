from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.polarquant import (
    recursive_polar_encode,
)
from turboquant.polarquant_quant import (
    DEFAULT_POLAR_BITS_BY_LEVEL,
    fit_polar_angle_codebooks_from_encodings,
)
from turboquant.polar_prod import (
    turboquant_polar_prod_quantize,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)
from turboquant.turboquant_logits_cuda import (
    turboquant_fused_logits_cuda,
)
from turboquant.packed_meta import (
    build_turboquant_packed_meta_blob,
)


# ============================================================
# Utilities
# ============================================================

def _ensure_cuda(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch_device = torch.device(device)

    if torch_device.type != "cuda":
        raise ValueError(
            f"This benchmark expects a CUDA device, got {device!r}."
        )

    return torch_device


def _sync() -> None:
    torch.cuda.synchronize()


@torch.no_grad()
def _bench_cuda_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}.")

    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}.")

    out: torch.Tensor | None = None

    for _ in range(warmup):
        out = fn()

    _sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        out = fn()

    end.record()

    _sync()

    elapsed_ms = float(
        start.elapsed_time(end)
    )

    if out is None:
        raise RuntimeError("Benchmark function did not produce output.")

    return elapsed_ms / float(iters), out


def _tensor_storage_bytes(x: torch.Tensor) -> int:
    return int(
        x.numel()
        * x.element_size()
    )


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0.0:
        return float("inf")
    return numerator / denominator


@torch.no_grad()
def _score_error_metrics(
    *,
    scores_ref: torch.Tensor,
    scores_test: torch.Tensor,
) -> dict[str, float]:
    ref = scores_ref.to(torch.float32)
    test = scores_test.to(torch.float32)

    if tuple(ref.shape) != tuple(test.shape):
        raise ValueError(
            "Score shape mismatch: "
            f"ref={tuple(ref.shape)}, test={tuple(test.shape)}."
        )

    err = test - ref
    abs_err = err.abs()

    mae = float(
        abs_err.mean().item()
    )

    rmse = float(
        torch.sqrt(
            torch.mean(
                err * err
            )
        ).item()
    )

    ref_abs_mean = float(
        ref.abs().mean().item()
    )

    ref_rms = float(
        torch.sqrt(
            torch.mean(
                ref * ref
            )
        ).item()
    )

    relative_mae = _safe_ratio(
        mae,
        ref_abs_mean,
    )

    relative_rmse = _safe_ratio(
        rmse,
        ref_rms,
    )

    max_abs_error = float(
        abs_err.max().item()
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "relative_mae": relative_mae,
        "relative_rmse": relative_rmse,
        "max_abs_error": max_abs_error,
    }


def _format_int(x: int) -> str:
    return f"{x:,}"


# ============================================================
# Shared calibration objects
# ============================================================

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


# ============================================================
# One sequence length benchmark
# ============================================================

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
        raise ValueError(
            f"seq_len must be > 0, got {seq_len}."
        )

    torch.manual_seed(
        seed + int(seq_len)
    )

    torch.cuda.manual_seed_all(
        seed + int(seq_len)
    )

    # ------------------------------------------------------------
    # Dense FP32 tensors
    # ------------------------------------------------------------

    q = torch.randn(
        B,
        H,
        Q,
        D,
        device=device,
        dtype=torch.float32,
    )

    k_dense = torch.randn(
        B,
        H,
        seq_len,
        D,
        device=device,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------
    # TurboQuant compressed K
    # ------------------------------------------------------------

    encoding = turboquant_polar_prod_quantize(
        x=k_dense,
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
        q.to(torch.float32),
        sketch.T.to(torch.float32),
    ).contiguous()

    # ------------------------------------------------------------
    # Dense logits reference
    # ------------------------------------------------------------

    def dense_fp32_logits() -> torch.Tensor:
        return torch.matmul(
            q,
            k_dense.transpose(-1, -2),
        )

    # ------------------------------------------------------------
    # TurboQuant fused logits
    # ------------------------------------------------------------

    def turboquant_logits() -> torch.Tensor:
        return turboquant_fused_logits_cuda(
            q=q,
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

    dense_ms, dense_scores = _bench_cuda_ms(
        dense_fp32_logits,
        warmup=warmup,
        iters=iters,
    )

    tq_ms, tq_scores = _bench_cuda_ms(
        turboquant_logits,
        warmup=warmup,
        iters=iters,
    )

    # ------------------------------------------------------------
    # Quality vs dense FP32 logits
    # ------------------------------------------------------------

    metrics = _score_error_metrics(
        scores_ref=dense_scores,
        scores_test=tq_scores,
    )

    # ------------------------------------------------------------
    # Storage accounting
    # ------------------------------------------------------------

    dense_fp32_k_bytes = _tensor_storage_bytes(
        k_dense
    )

    # Logical storage used by original packed representation:
    #   L1 32 B
    #   L2  8 B
    #   L3  4 B
    #   L4  2 B
    #   QJL signs 16 B
    #   radii 8 fp16 = 16 B
    #   residual norm 1 fp16 = 2 B
    #
    # Total = 80 B / 128 channels = 5.0 bpc.
    logical_tq_k_bytes = (
        _tensor_storage_bytes(
            packed.level1_4bit
        )
        + _tensor_storage_bytes(
            packed.level2_2bit
        )
        + _tensor_storage_bytes(
            packed.level3_2bit
        )
        + _tensor_storage_bytes(
            packed.level4_2bit
        )
        + _tensor_storage_bytes(
            packed_qjl_signs
        )
        + _tensor_storage_bytes(
            encoding.polar.radii
        )
        + _tensor_storage_bytes(
            qjl_norms
        )
    )

    # Physical storage if using meta64 as the actual stored format:
    #   packed_meta 64 B
    #   radii       16 B
    #   norm         2 B
    #
    # Total = 82 B / 128 channels = 5.125 bpc.
    meta64_physical_tq_k_bytes = (
        _tensor_storage_bytes(
            packed_meta
        )
        + _tensor_storage_bytes(
            encoding.polar.radii
        )
        + _tensor_storage_bytes(
            qjl_norms
        )
    )

    num_k_channels = (
        B
        * H
        * seq_len
        * D
    )

    logical_effective_k_bpc = (
        float(logical_tq_k_bytes) * 8.0
        / float(num_k_channels)
    )

    meta64_physical_k_bpc = (
        float(meta64_physical_tq_k_bytes) * 8.0
        / float(num_k_channels)
    )

    result = {
        "seq_len": int(seq_len),

        "dense_fp32_logits_ms": float(dense_ms),
        "turboquant_logits_ms": float(tq_ms),
        "speedup_dense_fp32_over_turboquant": float(
            dense_ms / tq_ms
        ),

        "dense_fp32_k_bytes": int(dense_fp32_k_bytes),

        "turboquant_logical_k_bytes": int(
            logical_tq_k_bytes
        ),
        "turboquant_logical_k_storage_ratio": float(
            logical_tq_k_bytes
            / dense_fp32_k_bytes
        ),
        "effective_tq_logical_k_bpc": float(
            logical_effective_k_bpc
        ),

        "turboquant_meta64_physical_k_bytes": int(
            meta64_physical_tq_k_bytes
        ),
        "turboquant_meta64_physical_k_storage_ratio": float(
            meta64_physical_tq_k_bytes
            / dense_fp32_k_bytes
        ),
        "effective_tq_meta64_physical_k_bpc": float(
            meta64_physical_k_bpc
        ),

        "quality_vs_dense_fp32_logits": metrics,
    }

    # ------------------------------------------------------------
    # Print one block
    # ------------------------------------------------------------

    print(
        f"dense FP32 logits:      "
        f"{dense_ms:.6f} ms"
    )

    print(
        f"TurboQuant logits:      "
        f"{tq_ms:.6f} ms"
    )

    print(
        f"speedup:                "
        f"{dense_ms / tq_ms:.4f}x"
    )

    print()

    print(
        f"dense FP32 K bytes:     "
        f"{_format_int(dense_fp32_k_bytes)}"
    )

    print(
        f"TurboQuant K bytes:     "
        f"{_format_int(logical_tq_k_bytes)}"
    )

    print(
        f"K storage ratio:        "
        f"{logical_tq_k_bytes / dense_fp32_k_bytes:.6f}"
    )

    print(
        f"effective TQ K bpc:     "
        f"{logical_effective_k_bpc:.6f}"
    )

    print()

    print(
        f"meta64 physical K bytes:"
        f" {_format_int(meta64_physical_tq_k_bytes)}"
    )

    print(
        f"meta64 storage ratio:   "
        f"{meta64_physical_tq_k_bytes / dense_fp32_k_bytes:.6f}"
    )

    print(
        f"meta64 physical K bpc:  "
        f"{meta64_physical_k_bpc:.6f}"
    )

    print()

    print(
        "quality vs dense FP32 logits:"
    )

    print(
        f"  MAE:                  "
        f"{metrics['mae']:.6e}"
    )

    print(
        f"  RMSE:                 "
        f"{metrics['rmse']:.6e}"
    )

    print(
        f"  relative RMSE:        "
        f"{metrics['relative_rmse']:.6e}"
    )

    print(
        f"  max abs error:        "
        f"{metrics['max_abs_error']:.6e}"
    )

    # ------------------------------------------------------------
    # Explicitly release large tensors before next sequence length
    # ------------------------------------------------------------

    del q
    del k_dense
    del encoding
    del packed
    del packed_qjl_signs
    del qjl_norms
    del packed_meta
    del q_projected
    del dense_scores
    del tq_scores

    torch.cuda.empty_cache()

    return result


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-style TurboQuant attention-logits benchmark: "
            "dense FP32 qK^T vs TurboQuant fused logits."
        )
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
        "--out",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
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
        "--num_queries",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--head_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--qjl_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_levels",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n_calib",
        type=int,
        default=4096,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = _ensure_cuda(
        args.device
    )

    B = int(args.batch_size)
    H = int(args.num_heads)
    Q = int(args.num_queries)
    D = int(args.head_dim)
    M = int(args.qjl_dim)
    num_levels = int(args.num_levels)

    if D != 128:
        raise ValueError(
            "Current fused CUDA kernel expects head_dim D=128, "
            f"got D={D}."
        )

    if M != 128:
        raise ValueError(
            "Current fused CUDA kernel expects QJL M=128, "
            f"got M={M}."
        )

    print(
        "========== Paper-style TurboQuant attention-logits benchmark =========="
    )

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
        "[Note] This aligns the metric with the public TurboQuant "
        "attention-logits speedup claim: dense FP32 qK^T vs "
        "TurboQuant logits. The logical compressed K bit-width is "
        "reported explicitly and is not silently labeled as 4-bit. "
        "For the meta64 fast path, physical storage including 2-byte "
        "padding is also reported."
    )

    codebooks, sketch = build_codebooks_and_sketch(
        device=device,
        d=D,
        m=M,
        num_levels=num_levels,
        n_calib=int(args.n_calib),
        seed=int(args.seed),
    )

    results: list[dict[str, Any]] = []

    for seq_len in args.seq_lens:
        print()
        print("=" * 72)
        print(
            f"[Attention-logits benchmark] T={int(seq_len)}"
        )
        print("=" * 72)

        result = benchmark_one_length(
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

        results.append(
            result
        )

    out_path = Path(
        args.out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "benchmark": "turboquant_paper_attention_logits_speedup",
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
            "seq_lens": [
                int(x)
                for x in args.seq_lens
            ],
            "seed": int(args.seed),
            "n_calib": int(args.n_calib),
            "bits_by_level": list(
                DEFAULT_POLAR_BITS_BY_LEVEL
            ),
        },
        "notes": {
            "metric_alignment": (
                "dense FP32 qK^T vs TurboQuant fused attention logits"
            ),
            "logical_compressed_k": (
                "logical storage excludes the 2-byte padding used by "
                "the meta64 fast-path layout"
            ),
            "meta64_physical_k": (
                "physical storage if packed_meta[64] replaces split "
                "packed metadata tensors"
            ),
        },
        "results": results,
    }

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print()
    print(
        f"[Save] {out_path}"
    )

    print(
        "[PASS] Paper-style TurboQuant attention-logits benchmark completed."
    )


if __name__ == "__main__":
    main()