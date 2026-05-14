from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import repeat_kv

from turboquant.llama_score_patch import (
    patch_llama_model_with_turboquant_scores,
)
from turboquant.polar_runtime_cache import (
    TurboQuantPolarRuntimeCache,
)
from turboquant.polarquant import (
    recursive_polar_encode,
)
from turboquant.polarquant_quant import (
    DEFAULT_POLAR_BITS_BY_LEVEL,
    fit_polar_angle_codebooks_from_encodings,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)


# ============================================================
# Utilities
# ============================================================

@contextmanager
def _nvtx_range(
    name: str,
    enabled: bool,
):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def _bytes_to_gb(x: int) -> float:
    return float(x) / (1024 ** 3)


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _json_dump(
    obj: dict[str, Any],
    path: str,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _clone_input_ids(
    input_ids: torch.Tensor,
) -> torch.Tensor:
    return input_ids.detach().clone()


def _build_long_prompt_ids(
    tokenizer,
    *,
    prompt_len: int,
    device: str,
) -> torch.Tensor:
    """
    Build a deterministic prompt with exactly prompt_len tokens.

    We start with a natural sentence and tile token ids until the
    requested length is reached.
    """
    text = (
        "TurboQuant is useful for language model inference because "
        "KV cache memory and attention computation become expensive "
        "at long context lengths. "
    )

    seed_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"][0]

    if int(seed_ids.numel()) == 0:
        raise RuntimeError("Tokenizer produced an empty prompt.")

    repeats = math.ceil(prompt_len / int(seed_ids.numel()))
    ids = seed_ids.repeat(repeats)[:prompt_len]

    return ids.unsqueeze(0).to(device)


def _past_key_values_dense_kv_bytes(
    past_key_values: Any,
) -> int:
    """
    Legacy tuple/list style HF past_key_values:
      layer -> (K,V)
    Also tolerates Cache-like structures if iterable.

    Returns total K+V bytes.
    """
    if past_key_values is None:
        return 0

    total = 0

    try:
        layers = list(past_key_values)
    except Exception:
        return 0

    for layer_cache in layers:
        if layer_cache is None:
            continue

        try:
            k, v = layer_cache[0], layer_cache[1]
        except Exception:
            continue

        if torch.is_tensor(k):
            total += k.numel() * k.element_size()
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()

    return int(total)


def _past_key_values_dense_cache_seq_len(
    past_key_values: Any,
) -> int:
    if past_key_values is None:
        return 0

    try:
        first = past_key_values[0]
        k = first[0]
        return int(k.shape[-2])
    except Exception:
        return 0


def _extract_generated_token_ids(
    token_steps: list[int],
) -> list[int]:
    return [int(x) for x in token_steps]


def _safe_token_prefix_suffix(
    tokens: list[int],
    n: int = 16,
) -> tuple[list[int], list[int]]:
    if len(tokens) <= n:
        return tokens, tokens
    return tokens[:n], tokens[-n:]


# ============================================================
# Model loading
# ============================================================

def _load_model_and_tokenizer(
    *,
    model_path: str,
    device: str,
):
    """
    This assumes model_path is the same SVD model path already used
    by your existing benchmark workflow.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=None,
    )

    model.eval()
    model.to(device)

    return model, tokenizer


# ============================================================
# Manual decode loop: baseline fp16 KV
# ============================================================

@torch.no_grad()
def run_baseline_fp16_kv(
    *,
    model,
    input_ids: torch.Tensor,
    decode_steps: int,
    enable_nvtx: bool,
) -> dict[str, Any]:
    device = str(input_ids.device)

    # ------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------
    _clear_cuda()
    _reset_peak_memory()

    with _nvtx_range("baseline_dense_prefill", enable_nvtx):
        _sync()
        t0 = time.perf_counter()

        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )

        _sync()
        t1 = time.perf_counter()

    prefill_latency_sec = float(t1 - t0)
    prefill_peak_cuda_memory_bytes = _peak_memory_bytes()

    past = outputs.past_key_values
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    prefill_cache_seq_len = _past_key_values_dense_cache_seq_len(
        past
    )
    prefill_dense_kv_bytes = _past_key_values_dense_kv_bytes(
        past
    )

    generated: list[int] = []

    # ------------------------------------------------------------
    # Decode loop
    # ------------------------------------------------------------
    _clear_cuda()
    _reset_peak_memory()

    with _nvtx_range("baseline_decode_loop", enable_nvtx):
        _sync()
        t2 = time.perf_counter()

        for step in range(decode_steps):
            with _nvtx_range(
                f"baseline_decode_loop_step_{step}",
                enable_nvtx,
            ):
                token_id = int(next_token.item())
                generated.append(token_id)

                outputs = model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                )

                past = outputs.past_key_values

                next_token = torch.argmax(
                    outputs.logits[:, -1, :],
                    dim=-1,
                    keepdim=True,
                )

        _sync()
        t3 = time.perf_counter()

    decode_latency_sec = float(t3 - t2)
    decode_peak_cuda_memory_bytes = _peak_memory_bytes()
    decode_tok_per_sec = (
        float(decode_steps) / decode_latency_sec
        if decode_latency_sec > 0
        else float("inf")
    )

    final_dense_cache_seq_len = _past_key_values_dense_cache_seq_len(
        past
    )
    final_dense_kv_bytes = _past_key_values_dense_kv_bytes(
        past
    )

    prefix, suffix = _safe_token_prefix_suffix(generated)

    return {
        "prefill_latency_sec": prefill_latency_sec,
        "prefill_peak_cuda_memory_bytes": int(
            prefill_peak_cuda_memory_bytes
        ),
        "prefill_peak_cuda_memory_gb": _bytes_to_gb(
            prefill_peak_cuda_memory_bytes
        ),
        "prefill_cache_seq_len": int(prefill_cache_seq_len),
        "prefill_dense_kv_bytes": int(prefill_dense_kv_bytes),

        "decode_latency_sec": decode_latency_sec,
        "decode_tok_per_sec": float(decode_tok_per_sec),
        "decode_peak_cuda_memory_bytes": int(
            decode_peak_cuda_memory_bytes
        ),
        "decode_peak_cuda_memory_gb": _bytes_to_gb(
            decode_peak_cuda_memory_bytes
        ),

        "generated_token_ids_prefix": prefix,
        "generated_token_ids_suffix": suffix,

        "final_dense_cache_seq_len": int(
            final_dense_cache_seq_len
        ),
        "final_dense_kv_bytes": int(
            final_dense_kv_bytes
        ),
    }


# ============================================================
# Polar TurboQuant cache setup
# ============================================================

@torch.no_grad()
def build_polar_runtime_cache(
    *,
    model,
    device: str,
    num_levels: int,
    qjl_m: int,
    polar_calib_samples: int,
    polar_calib_seed: int,
) -> TurboQuantPolarRuntimeCache:
    head_dim = (
        int(model.config.hidden_size)
        // int(model.config.num_attention_heads)
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(int(polar_calib_seed))

    x_calib = torch.randn(
        polar_calib_samples,
        head_dim,
        device=device,
        dtype=torch.float32,
        generator=gen,
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
        seed=polar_calib_seed,
    )

    sketch = make_gaussian_sketch(
        d=head_dim,
        m=qjl_m,
        device=device,
        dtype=torch.float32,
        seed=polar_calib_seed + 123,
    )

    return TurboQuantPolarRuntimeCache(
        num_layers=int(model.config.num_hidden_layers),
        codebooks=codebooks,
        sketch=sketch,
        num_levels=num_levels,
    )
    
@torch.no_grad()
def hydrate_polar_runtime_cache_from_dense_past(
    *,
    polar_cache: TurboQuantPolarRuntimeCache,
    dense_past_key_values,
    model,
) -> None:
    """
    Convert dense fp16 prefill KV cache into Polar TurboQuant runtime cache.

    Dense prefill is done with ordinary LLaMA attention.
    Then:
      - K is compressed into Polar/QJL cache
      - V is copied into dense V cache

    This avoids building an enormous [B,H,Q,T] Polar score matrix during prefill.
    """
    num_kv_groups = (
        int(model.config.num_attention_heads)
        // int(model.config.num_key_value_heads)
    )

    # Convert to mutable list so each layer's dense KV can be released
    # immediately after hydration.
    dense_layers = list(dense_past_key_values)

    for layer_idx in range(len(dense_layers)):
        layer_k, layer_v = dense_layers[layer_idx]

        # HF dense cache may be [B,num_kv_heads,T,D].
        # Custom Polar runtime score path expects full attention-head layout.
        if num_kv_groups > 1:
            layer_k = repeat_kv(
                layer_k,
                num_kv_groups,
            )
            layer_v = repeat_kv(
                layer_v,
                num_kv_groups,
            )

        polar_cache.update(
            layer_k,
            layer_v,
            layer_idx=layer_idx,
        )

        # Release this dense layer immediately.
        dense_layers[layer_idx] = None
        del layer_k
        del layer_v

    del dense_layers


def _polar_report_value_cache_bytes(
    report: dict[str, Any],
) -> int:
    return int(report.get("fp_value_cache_bytes", 0))

def _polar_report_compressed_k_bytes(
    report: dict[str, Any],
) -> int:
    return int(report.get("compressed_k_storage_bytes", 0))

def _polar_report_seq_len(
    report: dict[str, Any],
) -> int:
    return int(report.get("seen_tokens", 0))


# ============================================================
# Manual decode loop: Polar TurboQuant runtime cache
# ============================================================

@torch.no_grad()
def run_polar_turboquant_cache(
    *,
    model,
    input_ids: torch.Tensor,
    decode_steps: int,
    enable_nvtx: bool,
    num_levels: int,
    qjl_m: int,
    polar_calib_samples: int,
    polar_calib_seed: int,
) -> dict[str, Any]:
    device = str(input_ids.device)

    polar_cache = build_polar_runtime_cache(
        model=model,
        device=device,
        num_levels=num_levels,
        qjl_m=qjl_m,
        polar_calib_samples=polar_calib_samples,
        polar_calib_seed=polar_calib_seed,
    )

    # ------------------------------------------------------------
    # Dense prefill first.
    # Do NOT patch the model before this block.
    # ------------------------------------------------------------
    _clear_cuda()
    _reset_peak_memory()

    with _nvtx_range(
        "polar_dense_prefill_before_hydrate",
        enable_nvtx,
    ):
        _sync()
        t0 = time.perf_counter()

        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )

        _sync()
        t1 = time.perf_counter()

    prefill_latency_sec = float(t1 - t0)
    prefill_peak_cuda_memory_bytes = _peak_memory_bytes()

    dense_prefill_past = outputs.past_key_values

    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    # ------------------------------------------------------------
    # Patch only AFTER dense prefill.
    # Decode will use Polar runtime cache.
    # ------------------------------------------------------------
    patch_llama_model_with_turboquant_scores(
        model
    )

    # ------------------------------------------------------------
    # Hydrate Polar compressed K + dense V cache
    # from dense prefill KV.
    # ------------------------------------------------------------
    with _nvtx_range(
        "polar_cache_hydrate_from_dense_prefill",
        enable_nvtx,
    ):
        hydrate_polar_runtime_cache_from_dense_past(
            polar_cache=polar_cache,
            dense_past_key_values=dense_prefill_past,
            model=model,
        )

    del dense_prefill_past
    del outputs
    _clear_cuda()

    prefill_report = polar_cache.report()
    prefill_cache_seq_len = _polar_report_seq_len(
        prefill_report
    )

    generated: list[int] = []

    # ------------------------------------------------------------
    # Decode loop
    # ------------------------------------------------------------
    _clear_cuda()
    _reset_peak_memory()

    with _nvtx_range("polar_tq_decode_loop", enable_nvtx):
        _sync()
        t2 = time.perf_counter()

        for step in range(decode_steps):
            with _nvtx_range(
                f"polar_tq_decode_loop_step_{step}",
                enable_nvtx,
            ):
                token_id = int(next_token.item())
                generated.append(token_id)

                outputs = model(
                    input_ids=next_token,
                    past_key_values=polar_cache,
                    use_cache=True,
                )

                next_token = torch.argmax(
                    outputs.logits[:, -1, :],
                    dim=-1,
                    keepdim=True,
                )

        _sync()
        t3 = time.perf_counter()

    decode_latency_sec = float(t3 - t2)
    decode_peak_cuda_memory_bytes = _peak_memory_bytes()
    decode_tok_per_sec = (
        float(decode_steps) / decode_latency_sec
        if decode_latency_sec > 0
        else float("inf")
    )

    final_report = polar_cache.report()
    final_cache_seq_len = _polar_report_seq_len(
        final_report
    )

    final_fp16_v_bytes = _polar_report_value_cache_bytes(
        final_report
    )
    
    final_compressed_k_bytes = _polar_report_compressed_k_bytes(
        final_report
    )

    final_total_cache_bytes = (
        final_compressed_k_bytes +
        final_fp16_v_bytes
    )

    prefix, suffix = _safe_token_prefix_suffix(generated)

    return {
        "prefill_latency_sec": prefill_latency_sec,
        "prefill_peak_cuda_memory_bytes": int(
            prefill_peak_cuda_memory_bytes
        ),
        "prefill_peak_cuda_memory_gb": _bytes_to_gb(
            prefill_peak_cuda_memory_bytes
        ),
        "prefill_cache_seq_len": int(prefill_cache_seq_len),

        "decode_latency_sec": decode_latency_sec,
        "decode_tok_per_sec": float(decode_tok_per_sec),
        "decode_peak_cuda_memory_bytes": int(
            decode_peak_cuda_memory_bytes
        ),
        "decode_peak_cuda_memory_gb": _bytes_to_gb(
            decode_peak_cuda_memory_bytes
        ),

        "generated_token_ids_prefix": prefix,
        "generated_token_ids_suffix": suffix,

        "final_polar_cache_seq_len": int(final_cache_seq_len),
        "final_compressed_k_bytes": int(
            final_compressed_k_bytes
        ),
        "final_fp16_v_bytes": int(
            final_fp16_v_bytes
        ),
        "final_total_cache_bytes": int(
            final_total_cache_bytes
        ),

        "runtime_cache_report": final_report,
    }


# ============================================================
# Summary
# ============================================================

def build_summary(
    *,
    baseline: dict[str, Any],
    polar: dict[str, Any],
) -> dict[str, Any]:
    baseline_tok_s = float(
        baseline["decode_tok_per_sec"]
    )

    polar_tok_s = float(
        polar["decode_tok_per_sec"]
    )

    decode_speed_ratio_polar_over_fp16 = (
        polar_tok_s / baseline_tok_s
        if baseline_tok_s > 0
        else float("nan")
    )

    decode_latency_speedup_fp16_over_polar = (
        float(baseline["decode_latency_sec"])
        / float(polar["decode_latency_sec"])
        if float(polar["decode_latency_sec"]) > 0
        else float("nan")
    )

    baseline_final_dense_kv_bytes = int(
        baseline["final_dense_kv_bytes"]
    )

    polar_final_compressed_k_bytes = int(
        polar["final_compressed_k_bytes"]
    )

    polar_final_fp16_v_bytes = int(
        polar["final_fp16_v_bytes"]
    )

    polar_final_total_cache_bytes = int(
        polar["final_total_cache_bytes"]
    )

    overall_cache_ratio_polar_over_fp16 = (
        float(polar_final_total_cache_bytes)
        / float(baseline_final_dense_kv_bytes)
        if baseline_final_dense_kv_bytes > 0
        else float("nan")
    )

    overall_cache_reduction_percent = (
        1.0 - overall_cache_ratio_polar_over_fp16
    ) * 100.0

    return {
        "decode_latency_speedup_fp16_over_polar": float(
            decode_latency_speedup_fp16_over_polar
        ),
        "decode_throughput_ratio_polar_over_fp16": float(
            decode_speed_ratio_polar_over_fp16
        ),
        "baseline_decode_tok_per_sec": float(
            baseline_tok_s
        ),
        "polar_tq_decode_tok_per_sec": float(
            polar_tok_s
        ),

        "baseline_final_dense_kv_bytes": int(
            baseline_final_dense_kv_bytes
        ),
        "polar_final_compressed_k_bytes": int(
            polar_final_compressed_k_bytes
        ),
        "polar_final_fp16_v_bytes": int(
            polar_final_fp16_v_bytes
        ),
        "polar_final_total_cache_bytes": int(
            polar_final_total_cache_bytes
        ),
        "overall_cache_ratio_polar_over_fp16": float(
            overall_cache_ratio_polar_over_fp16
        ),
        "overall_cache_reduction_percent": float(
            overall_cache_reduction_percent
        ),
    }
# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="runs/svd_uniform_08/model/svd_uniform_08",
        help="Path to the SVD model/tokenizer used by existing benchmark workflow.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--prompt_len",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--decode_steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_levels",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--qjl_m",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--polar_calib_samples",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--polar_calib_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--enable_nvtx",
        action="store_true",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.out is None:
        args.out = (
            f"runs/svd_uniform_08/eval/"
            f"bench_turboquant_polar_long_decode_only_"
            f"p{args.prompt_len}_d{args.decode_steps}.json"
        )

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = args.device

    # ============================================================
    # Load first model for baseline
    # ============================================================

    print("========== Load baseline SVD model ==========")
    baseline_model, tokenizer = _load_model_and_tokenizer(
        model_path=args.model_path,
        device=device,
    )

    input_ids = _build_long_prompt_ids(
        tokenizer,
        prompt_len=args.prompt_len,
        device=device,
    )

    print("========== A. Baseline SVD + fp16 KV ==========")
    baseline = run_baseline_fp16_kv(
        model=baseline_model,
        input_ids=_clone_input_ids(input_ids),
        decode_steps=args.decode_steps,
        enable_nvtx=args.enable_nvtx,
    )
    print(json.dumps(baseline, indent=2))

    del baseline_model
    _clear_cuda()

    # ============================================================
    # Load fresh model for Polar TurboQuant path
    # ============================================================

    print()
    print("========== Load Polar TurboQuant SVD model ==========")
    polar_model, _ = _load_model_and_tokenizer(
        model_path=args.model_path,
        device=device,
    )

    print("========== B. Polar TurboQuant runtime cache ==========")
    polar = run_polar_turboquant_cache(
        model=polar_model,
        input_ids=_clone_input_ids(input_ids),
        decode_steps=args.decode_steps,
        enable_nvtx=args.enable_nvtx,
        num_levels=args.num_levels,
        qjl_m=args.qjl_m,
        polar_calib_samples=args.polar_calib_samples,
        polar_calib_seed=args.polar_calib_seed,
    )
    print(json.dumps(polar, indent=2))

    del polar_model
    _clear_cuda()

    # ============================================================
    # Summary
    # ============================================================

    summary = build_summary(
        baseline=baseline,
        polar=polar,
    )

    result = {
        "config": {
            "model_path": args.model_path,
            "device": args.device,
            "prompt_len": int(args.prompt_len),
            "decode_steps": int(args.decode_steps),
            "num_levels": int(args.num_levels),
            "qjl_m": int(args.qjl_m),
            "polar_calib_samples": int(args.polar_calib_samples),
            "polar_calib_seed": int(args.polar_calib_seed),
            "enable_nvtx": bool(args.enable_nvtx),
        },
        "baseline": baseline,
        "polar_turboquant": polar,
        "summary": summary,
    }

    print()
    print("========== Summary ==========")
    print(json.dumps(summary, indent=2))

    _json_dump(
        result,
        args.out,
    )

    print()
    print(f"[Save] {args.out}")
    print("[PASS] Polar TurboQuant long-context decode-only benchmark completed.")


if __name__ == "__main__":
    main()