#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.polarquant import (
    PolarEncoding,
    recursive_polar_decode,
    recursive_polar_encode,
)
from turboquant.polarquant_quant import (
    fit_polar_angle_codebooks_from_encodings,
    quantize_angle_tensor,
)
from turboquant.qjl import (
    make_gaussian_sketch,
    qjl_encode,
)
from turboquant.polar_packing_3bpc import (
    PackedPolarAngles3Bpc,
    pack_polar_angle_codes_3bpc_l4_d128,
)
from turboquant.packed_meta_3bpc import (
    build_polarquant_3bpc_packed_meta32_blob,
)
from turboquant.turboquant_logits_3bpc_cuda import (
    polarquant_3bpc_fused_logits_cuda,
)


BITS_BY_LEVEL_3BPC = (2, 1, 1, 1)
D = 128
M = 64
NUM_LEVELS = 4


def _sync() -> None:
    torch.cuda.synchronize()


def _storage_bytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())


@torch.no_grad()
def _bench_cuda_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
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

    if out is None:
        raise RuntimeError("Benchmark function did not produce output.")

    return float(start.elapsed_time(end) / iters), out


def _safe_ratio(x: float, y: float) -> float:
    if y == 0.0:
        return float("inf")
    return x / y


@torch.no_grad()
def _score_error_metrics(
    *,
    scores_ref: torch.Tensor,
    scores_test: torch.Tensor,
) -> dict[str, float]:
    ref = scores_ref.to(torch.float32)
    test = scores_test.to(torch.float32)

    err = test - ref
    abs_err = err.abs()

    mae = float(abs_err.mean().item())
    rmse = float(torch.sqrt(torch.mean(err * err)).item())

    ref_abs_mean = float(ref.abs().mean().item())
    ref_rms = float(torch.sqrt(torch.mean(ref * ref)).item())

    return {
        "mae": mae,
        "rmse": rmse,
        "relative_mae": _safe_ratio(mae, ref_abs_mean),
        "relative_rmse": _safe_ratio(rmse, ref_rms),
        "max_abs_error": float(abs_err.max().item()),
    }


@torch.no_grad()
def build_codebooks_and_sketch(
    *,
    device: torch.device,
    n_calib: int,
    seed: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x_calib = torch.randn(
        n_calib,
        D,
        device=device,
        dtype=torch.float32,
    )

    enc_calib = recursive_polar_encode(
        x_calib,
        num_levels=NUM_LEVELS,
    )

    codebooks = fit_polar_angle_codebooks_from_encodings(
        [enc_calib],
        bits_by_level=BITS_BY_LEVEL_3BPC,
        max_iters=30,
        max_samples_per_level=200_000,
        seed=seed,
    )

    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=seed + 123,
    )

    del x_calib
    del enc_calib
    torch.cuda.empty_cache()

    return codebooks, sketch


@torch.no_grad()
def _dequantize_chunk_from_codes(
    *,
    angle_codes: list[torch.Tensor],
    radii_store: torch.Tensor,
    codebooks,
) -> torch.Tensor:
    angles = [
        centroids[codes].to(torch.float32)
        for codes, centroids in zip(angle_codes, codebooks.centroids)
    ]

    polar = PolarEncoding(
        angles=angles,
        radii=radii_store.to(torch.float32),
        original_dim=D,
        num_levels=NUM_LEVELS,
    )

    return recursive_polar_decode(polar).to(torch.float32)


@torch.no_grad()
def build_3bpc_cache_tensors(
    *,
    k_dense_fp32: torch.Tensor,
    codebooks,
    sketch: torch.Tensor,
    build_chunk_size: int,
) -> dict[str, torch.Tensor]:
    B, H, T, D_seen = k_dense_fp32.shape
    if D_seen != D:
        raise ValueError(f"Expected D={D}, got D={D_seen}.")

    device = k_dense_fp32.device

    packed_l1 = torch.empty((B, H, T, 16), dtype=torch.uint8, device=device)
    packed_l2 = torch.empty((B, H, T, 4), dtype=torch.uint8, device=device)
    packed_l3 = torch.empty((B, H, T, 2), dtype=torch.uint8, device=device)
    packed_l4 = torch.empty((B, H, T, 1), dtype=torch.uint8, device=device)
    radii = torch.empty((B, H, T, 8), dtype=torch.float16, device=device)

    packed_qjl_signs = torch.empty((B, H, T, M // 8), dtype=torch.uint8, device=device)
    qjl_norms = torch.empty((B, H, T), dtype=torch.float16, device=device)

    for begin in range(0, T, build_chunk_size):
        end = min(begin + build_chunk_size, T)
        k_chunk = k_dense_fp32[:, :, begin:end, :].contiguous()

        polar_exact = recursive_polar_encode(
            k_chunk,
            num_levels=NUM_LEVELS,
        )

        angle_codes = [
            quantize_angle_tensor(
                angle=angle,
                centroids=centroids,
            )
            for angle, centroids in zip(
                polar_exact.angles,
                codebooks.centroids,
            )
        ]

        packed_chunk: PackedPolarAngles3Bpc = (
            pack_polar_angle_codes_3bpc_l4_d128(angle_codes)
        )

        radii_store = polar_exact.radii.to(torch.float16).contiguous()
        x_hat = _dequantize_chunk_from_codes(
            angle_codes=angle_codes,
            radii_store=radii_store,
            codebooks=codebooks,
        )

        residual_flat = (
            k_chunk.to(torch.float32)
            - x_hat.to(torch.float32)
        ).reshape(-1, D)

        qjl = qjl_encode(
            x=residual_flat,
            S=sketch,
        )

        t_chunk = end - begin

        packed_l1[:, :, begin:end, :].copy_(packed_chunk.level1_2bit)
        packed_l2[:, :, begin:end, :].copy_(packed_chunk.level2_1bit)
        packed_l3[:, :, begin:end, :].copy_(packed_chunk.level3_1bit)
        packed_l4[:, :, begin:end, :].copy_(packed_chunk.level4_1bit)
        radii[:, :, begin:end, :].copy_(radii_store)

        packed_qjl_signs[:, :, begin:end, :].copy_(
            qjl.packed_sign_bits.reshape(B, H, t_chunk, M // 8)
        )
        qjl_norms[:, :, begin:end].copy_(
            qjl.norms.reshape(B, H, t_chunk)
        )

        del k_chunk
        del polar_exact
        del angle_codes
        del packed_chunk
        del radii_store
        del x_hat
        del residual_flat
        del qjl

    packed_meta32 = build_polarquant_3bpc_packed_meta32_blob(
        packed_l1_2bit=packed_l1,
        packed_l2_1bit=packed_l2,
        packed_l3_1bit=packed_l3,
        packed_l4_1bit=packed_l4,
        packed_qjl_signs=packed_qjl_signs,
    )

    return {
        "packed_l1": packed_l1,
        "packed_l2": packed_l2,
        "packed_l3": packed_l3,
        "packed_l4": packed_l4,
        "radii": radii,
        "packed_qjl_signs": packed_qjl_signs,
        "qjl_norms": qjl_norms,
        "packed_meta32": packed_meta32,
    }


@torch.no_grad()
def benchmark_one_length(
    *,
    seq_len: int,
    B: int,
    H: int,
    device: torch.device,
    codebooks,
    sketch: torch.Tensor,
    warmup: int,
    iters: int,
    seed: int,
    build_chunk_size: int,
) -> dict[str, Any]:
    print()
    print("=" * 78)
    print(f"[PolarQuant aligned ~3bpc meta32 benchmark] T={seq_len}")
    print("=" * 78)

    torch.manual_seed(seed + seq_len)
    torch.cuda.manual_seed_all(seed + seq_len)

    q_fp32 = torch.randn(
        B, H, 1, D,
        device=device,
        dtype=torch.float32,
    )

    k_fp32 = torch.randn(
        B, H, seq_len, D,
        device=device,
        dtype=torch.float32,
    )

    q_fp16 = q_fp32.to(torch.float16)
    k_fp16 = k_fp32.to(torch.float16)

    cache = build_3bpc_cache_tensors(
        k_dense_fp32=k_fp32,
        codebooks=codebooks,
        sketch=sketch,
        build_chunk_size=build_chunk_size,
    )

    q_projected = torch.matmul(
        q_fp32,
        sketch.T.to(torch.float32),
    ).contiguous()

    def dense_fp16_einsum() -> torch.Tensor:
        return torch.einsum(
            "bhqd,bhkd->bhqk",
            q_fp16,
            k_fp16,
        )

    def dense_fp32_qkt() -> torch.Tensor:
        return torch.matmul(
            q_fp32,
            k_fp32.transpose(-1, -2),
        )

    def polarquant_3bpc_logits() -> torch.Tensor:
        return polarquant_3bpc_fused_logits_cuda(
            q=q_fp32,
            q_projected=q_projected,
            packed_meta32=cache["packed_meta32"],
            radii=cache["radii"],
            centroids_l1=codebooks.centroids[0],
            centroids_l2=codebooks.centroids[1],
            centroids_l3=codebooks.centroids[2],
            centroids_l4=codebooks.centroids[3],
            qjl_norms=cache["qjl_norms"],
        )

    dense_fp16_ms, _ = _bench_cuda_ms(
        dense_fp16_einsum,
        warmup=warmup,
        iters=iters,
    )
    dense_fp32_ms, dense_fp32_scores = _bench_cuda_ms(
        dense_fp32_qkt,
        warmup=warmup,
        iters=iters,
    )
    pq3_ms, pq3_scores = _bench_cuda_ms(
        polarquant_3bpc_logits,
        warmup=warmup,
        iters=iters,
    )

    quality = _score_error_metrics(
        scores_ref=dense_fp32_scores,
        scores_test=pq3_scores,
    )

    dense_fp16_k_bytes = _storage_bytes(k_fp16)
    dense_fp32_k_bytes = _storage_bytes(k_fp32)

    logical_angle_bytes = (
        _storage_bytes(cache["packed_l1"])
        + _storage_bytes(cache["packed_l2"])
        + _storage_bytes(cache["packed_l3"])
        + _storage_bytes(cache["packed_l4"])
    )
    logical_qjl_bytes = _storage_bytes(cache["packed_qjl_signs"])
    radii_bytes = _storage_bytes(cache["radii"])
    qjl_norm_bytes = _storage_bytes(cache["qjl_norms"])

    logical_k_bytes = (
        logical_angle_bytes
        + logical_qjl_bytes
        + radii_bytes
        + qjl_norm_bytes
    )

    physical_meta32_bytes = _storage_bytes(cache["packed_meta32"])
    physical_k_bytes = (
        physical_meta32_bytes
        + radii_bytes
        + qjl_norm_bytes
    )

    num_k_channels = B * H * seq_len * D
    logical_bpc = float(logical_k_bytes) * 8.0 / float(num_k_channels)
    physical_bpc = float(physical_k_bytes) * 8.0 / float(num_k_channels)

    result = {
        "seq_len": int(seq_len),
        "shape": {
            "B": int(B),
            "H": int(H),
            "Q": 1,
            "D": D,
            "M": M,
        },
        "aligned_3bpc_config": {
            "bits_by_level": list(BITS_BY_LEVEL_3BPC),
            "packed_meta_bytes_per_key": 32,
            "radii_bytes_per_key": 16,
            "qjl_norm_bytes_per_key": 2,
            "physical_bytes_per_key": 50,
            "physical_bpc_target": 3.125,
        },
        "timing_ms": {
            "dense_fp16_einsum_ms": float(dense_fp16_ms),
            "dense_fp32_qkt_ms": float(dense_fp32_ms),
            "polarquant_3bpc_meta32_fused_logits_ms": float(pq3_ms),
        },
        "speedup": {
            "over_dense_fp16_einsum": float(dense_fp16_ms / pq3_ms),
            "over_dense_fp32_qkt": float(dense_fp32_ms / pq3_ms),
        },
        "quality_vs_dense_fp32_qkt": quality,
        "memory_bytes": {
            "dense_fp16_k_bytes": int(dense_fp16_k_bytes),
            "dense_fp32_k_bytes": int(dense_fp32_k_bytes),
            "logical_3bpc_k_bytes": int(logical_k_bytes),
            "physical_meta32_3bpc_k_bytes": int(physical_k_bytes),
        },
        "memory_ratio": {
            "logical_over_dense_fp16_k": float(logical_k_bytes / dense_fp16_k_bytes),
            "logical_over_dense_fp32_k": float(logical_k_bytes / dense_fp32_k_bytes),
            "physical_over_dense_fp16_k": float(physical_k_bytes / dense_fp16_k_bytes),
            "physical_over_dense_fp32_k": float(physical_k_bytes / dense_fp32_k_bytes),
        },
        "effective_k_bits_per_channel": {
            "logical": float(logical_bpc),
            "physical_meta32": float(physical_bpc),
        },
    }

    print(json.dumps(result, indent=2))

    del q_fp32
    del k_fp32
    del q_fp16
    del k_fp16
    del cache
    del q_projected
    del dense_fp32_scores
    del pq3_scores
    torch.cuda.empty_cache()

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "PolarQuant aligned ~3bpc benchmark: "
            "meta32, QJL M=64, dual dense baselines."
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
    p.add_argument("--n_calib", type=int, default=4096)
    p.add_argument("--build_chunk_size", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"CUDA device required, got {args.device!r}.")

    B = int(args.batch_size)
    H = int(args.num_heads)

    if B != 1:
        raise ValueError("Current aligned 3bpc CUDA fast path requires B=1.")

    print("========== PolarQuant aligned ~3bpc meta32 benchmark ==========")
    print(f"device             = {device}")
    print(f"B                  = {B}")
    print(f"H                  = {H}")
    print(f"Q                  = 1")
    print(f"D                  = {D}")
    print(f"M                  = {M}")
    print(f"bits_by_level      = {BITS_BY_LEVEL_3BPC}")
    print(f"physical bpc target= 3.125")
    print(f"warmup             = {args.warmup}")
    print(f"iters              = {args.iters}")
    print(f"seq_lens           = {list(args.seq_lens)}")
    print()
    print(
        "[Alignment note] This experiment keeps the same paper-style "
        "attention-logits framing as the 5.125-bpc PolarQuant benchmark: "
        "dense FP16 einsum, dense FP32 qK^T, and a B=1/Q=1 fused logits fast path. "
        "The metadata blob is reduced from meta64 to aligned meta32."
    )

    codebooks, sketch = build_codebooks_and_sketch(
        device=device,
        n_calib=int(args.n_calib),
        seed=int(args.seed),
    )

    results = []
    for seq_len in args.seq_lens:
        results.append(
            benchmark_one_length(
                seq_len=int(seq_len),
                B=B,
                H=H,
                device=device,
                codebooks=codebooks,
                sketch=sketch,
                warmup=int(args.warmup),
                iters=int(args.iters),
                seed=int(args.seed),
                build_chunk_size=int(args.build_chunk_size),
            )
        )

    payload = {
        "benchmark": "polarquant_aligned_3bpc_meta32_dual_baseline_speedup",
        "method": "polarquant_base_meta32_m64_bits_2111",
        "device": str(device),
        "config": {
            "B": B,
            "H": H,
            "Q": 1,
            "D": D,
            "M": M,
            "num_levels": NUM_LEVELS,
            "bits_by_level": list(BITS_BY_LEVEL_3BPC),
            "physical_bpc_target": 3.125,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "seq_lens": [int(x) for x in args.seq_lens],
            "seed": int(args.seed),
            "n_calib": int(args.n_calib),
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
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"[Save] {out_path}")
    print("[PASS] PolarQuant aligned ~3bpc meta32 benchmark completed.")


if __name__ == "__main__":
    main()
