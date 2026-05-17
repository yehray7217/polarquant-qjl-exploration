from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.key_cache import TurboQuantKeyCache
from turboquant.mse_quant import (
    make_random_rotation,
    get_2bit_centroids,
)
from turboquant.qjl import make_gaussian_sketch
from turboquant.cuda_score import turboquant_decode_score_cuda_from_cache


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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

    elapsed_ms = float(start.elapsed_time(end)) / float(iters)
    return elapsed_ms, out


def _safe_ratio(x: float, y: float) -> float:
    if y == 0.0:
        return float("inf")
    return x / y


@torch.no_grad()
def _error_metrics(
    ref: torch.Tensor,
    test: torch.Tensor,
) -> dict[str, float]:
    ref_f = ref.to(torch.float32)
    test_f = test.to(torch.float32)

    err = test_f - ref_f
    abs_err = err.abs()

    mae = float(abs_err.mean().item())
    rmse = float(torch.sqrt(torch.mean(err * err)).item())

    ref_abs_mean = float(ref_f.abs().mean().item())
    ref_rms = float(torch.sqrt(torch.mean(ref_f * ref_f)).item())

    return {
        "mae": mae,
        "rmse": rmse,
        "relative_mae": _safe_ratio(mae, ref_abs_mean),
        "relative_rmse": _safe_ratio(rmse, ref_rms),
        "max_abs_error": float(abs_err.max().item()),
    }


@torch.no_grad()
def build_cache(
    *,
    B: int,
    H: int,
    T: int,
    D: int,
    M: int,
    device: str,
    seed: int,
    append_chunk_size: int,
) -> tuple[TurboQuantKeyCache, torch.Tensor]:
    torch.manual_seed(seed + T)
    torch.cuda.manual_seed_all(seed + T)

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

    key_states = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=torch.float16,
    )

    # Score latency only; compression/build time is excluded.
    for begin in range(0, T, append_chunk_size):
        end = min(begin + append_chunk_size, T)
        cache.append(
            layer_idx=0,
            key_states=key_states[:, :, begin:end, :],
            value_states=None,
        )

    query_states = torch.randn(
        B,
        H,
        1,
        D,
        device=device,
        dtype=torch.float16,
    )

    return cache, query_states


@torch.no_grad()
def benchmark_one_length(
    *,
    T: int,
    B: int,
    H: int,
    D: int,
    M: int,
    device: str,
    seed: int,
    python_warmup: int,
    python_iters: int,
    cuda_warmup: int,
    cuda_iters: int,
    append_chunk_size: int,
) -> dict[str, Any]:
    print()
    print("=" * 78)
    print(f"[Python score vs CUDA score] T={T}")
    print("=" * 78)

    cache, query_states = build_cache(
        B=B,
        H=H,
        T=T,
        D=D,
        M=M,
        device=device,
        seed=seed,
        append_chunk_size=append_chunk_size,
    )

    def python_score() -> torch.Tensor:
        return cache.score(
            layer_idx=0,
            query_states=query_states,
        )

    def cuda_score() -> torch.Tensor:
        return turboquant_decode_score_cuda_from_cache(
            query_states=query_states,
            cache=cache,
            layer_idx=0,
        )

    python_ms, python_out = _bench_cuda_ms(
        python_score,
        warmup=python_warmup,
        iters=python_iters,
    )

    cuda_ms, cuda_out = _bench_cuda_ms(
        cuda_score,
        warmup=cuda_warmup,
        iters=cuda_iters,
    )

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
            "python_reference_score_ms": float(python_ms),
            "cuda_decode_score_ms": float(cuda_ms),
        },
        "speedup": {
            "cuda_over_python_score_latency": float(python_ms / cuda_ms),
        },
        "quality_cuda_vs_python_score": _error_metrics(
            ref=python_out,
            test=cuda_out,
        ),
        "compressed_k_report": cache.report()["layers"][0],
        "notes": {
            "measured_region": "score path only; cache construction is excluded",
            "python_path": "TurboQuantKeyCache.score",
            "cuda_path": "turboquant_decode_score_cuda_from_cache",
        },
    }

    print(json.dumps(result, indent=2))

    del cache
    del query_states
    del python_out
    del cuda_out
    torch.cuda.empty_cache()

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reconstruct pre-CUDA vs CUDA TurboQuant score latency: "
            "Python reference score vs CUDA decode score."
        )
    )
    p.add_argument("--seq_lens", type=int, nargs="+", default=[2176])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--qjl_dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--append_chunk_size", type=int, default=1024)

    # The Python reference path is much heavier.
    p.add_argument("--python_warmup", type=int, default=2)
    p.add_argument("--python_iters", type=int, default=10)

    p.add_argument("--cuda_warmup", type=int, default=20)
    p.add_argument("--cuda_iters", type=int, default=100)

    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if not str(args.device).startswith("cuda"):
        raise ValueError(f"Expected CUDA device, got {args.device!r}.")

    results = []
    for T in args.seq_lens:
        results.append(
            benchmark_one_length(
                T=int(T),
                B=int(args.batch_size),
                H=int(args.num_heads),
                D=int(args.head_dim),
                M=int(args.qjl_dim),
                device=str(args.device),
                seed=int(args.seed),
                python_warmup=int(args.python_warmup),
                python_iters=int(args.python_iters),
                cuda_warmup=int(args.cuda_warmup),
                cuda_iters=int(args.cuda_iters),
                append_chunk_size=int(args.append_chunk_size),
            )
        )

    payload = {
        "benchmark": "turboquant_python_vs_cuda_score",
        "config": {
            "seq_lens": [int(x) for x in args.seq_lens],
            "device": str(args.device),
            "B": int(args.batch_size),
            "H": int(args.num_heads),
            "Q": 1,
            "D": int(args.head_dim),
            "M": int(args.qjl_dim),
            "seed": int(args.seed),
            "append_chunk_size": int(args.append_chunk_size),
            "python_warmup": int(args.python_warmup),
            "python_iters": int(args.python_iters),
            "cuda_warmup": int(args.cuda_warmup),
            "cuda_iters": int(args.cuda_iters),
        },
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"[Save] {out_path}")
    print("[PASS] Python-vs-CUDA TurboQuant score benchmark completed.")


if __name__ == "__main__":
    main()
