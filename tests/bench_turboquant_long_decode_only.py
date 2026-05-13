from pathlib import Path
import sys
import json
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.llama_score_patch import (
    patch_llama_model_with_turboquant_scores,
)
from turboquant.runtime_cache import TurboQuantRuntimeCache


# ============================================================
# Utilities
# ============================================================

def load_cached_svd_model(path: str, device: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model = obj["model"]
    tokenizer = obj["tokenizer"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    model.to(device)
    return model, tokenizer


def build_fixed_length_prompt(
    tokenizer,
    prompt_len: int,
    device: str,
):
    """
    Build a deterministic text prompt and truncate to exactly prompt_len tokens.
    """
    base_text = (
        "TurboQuant compresses key cache representations for efficient "
        "long-context autoregressive decoding in large language models. "
        "This sentence is repeated to construct a controlled benchmark prompt. "
    )

    text = base_text * 4096

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_len,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    if input_ids.shape[1] != prompt_len:
        raise RuntimeError(
            f"Failed to build exact prompt length {prompt_len}, "
            f"got {input_ids.shape[1]}"
        )

    return (
        input_ids.to(device),
        attention_mask.to(device),
    )


def dense_past_num_bytes(past_key_values) -> int:
    total = 0

    for layer in past_key_values:
        key_states, value_states = layer[0], layer[1]
        total += key_states.numel() * key_states.element_size()
        total += value_states.numel() * value_states.element_size()

    return int(total)


def dense_past_seq_len(past_key_values) -> int:
    return int(past_key_values[0][0].shape[-2])


def maybe_nvtx_push(enabled: bool, name: str):
    if enabled:
        torch.cuda.nvtx.range_push(name)


def maybe_nvtx_pop(enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_pop()


# ============================================================
# Dense prefill
# ============================================================

@torch.no_grad()
def prefill_dense_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    enable_nvtx: bool = False,
    nvtx_name: str = "dense_prefill",
):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    maybe_nvtx_push(enable_nvtx, nvtx_name)

    t0 = time.perf_counter()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    maybe_nvtx_pop(enable_nvtx)

    next_token = torch.argmax(
        out.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    peak_mem = torch.cuda.max_memory_allocated(device)

    return {
        "past_key_values": out.past_key_values,
        "next_token": next_token,
        "prefill_latency_sec": float(t1 - t0),
        "prefill_peak_cuda_memory_bytes": int(peak_mem),
        "prefill_peak_cuda_memory_gb": float(peak_mem / (1024 ** 3)),
        "prefill_cache_seq_len": dense_past_seq_len(out.past_key_values),
        "prefill_dense_kv_bytes": dense_past_num_bytes(out.past_key_values),
    }


# ============================================================
# Dense -> TurboQuant runtime cache hydration
# ============================================================

@torch.no_grad()
def hydrate_turboquant_runtime_cache_from_dense_past(
    dense_past,
    tq_state,
    num_layers: int,
    device: str,
    max_cache_len: int,
    enable_nvtx: bool = False,
):
    """
    Convert baseline dense K/V cache into:
      compressed K + fp16 V runtime cache.

    This conversion is timed separately and excluded from decode-only latency.
    """
    runtime_cache = TurboQuantRuntimeCache(
        num_layers=num_layers,
        rotation=tq_state.rotation,
        centroids=tq_state.centroids,
        sketch=tq_state.sketch,
        max_cache_len=max_cache_len,
    )

    torch.cuda.synchronize()

    maybe_nvtx_push(enable_nvtx, "hydrate_turboquant_runtime_cache")

    t0 = time.perf_counter()

    for layer_idx, layer in enumerate(dense_past):
        key_states, value_states = layer[0], layer[1]

        runtime_cache.update(
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            cache_kwargs={},
        )

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    maybe_nvtx_pop(enable_nvtx)

    return runtime_cache, float(t1 - t0)


# ============================================================
# Manual decode loop
# ============================================================

@torch.no_grad()
def manual_decode_loop(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    first_token: torch.Tensor,
    past_key_values,
    decode_steps: int,
    device: str,
    enable_nvtx: bool = False,
    nvtx_name: str = "decode_loop",
):
    """
    Run exactly `decode_steps` one-token forward passes after prefill.

    Note:
      - first_token is selected from prefill logits
      - each loop iteration forwards one token and predicts the next
      - throughput is measured as decode_steps / decode_latency
    """
    current_full_input_ids = torch.cat(
        [input_ids, first_token],
        dim=-1,
    )

    current_attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ],
        dim=-1,
    )

    past = past_key_values
    generated_tokens = [int(first_token[0, 0].item())]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    maybe_nvtx_push(enable_nvtx, nvtx_name)

    t0 = time.perf_counter()

    for step_idx in range(decode_steps):
        maybe_nvtx_push(enable_nvtx, f"{nvtx_name}_step_{step_idx}")

        model_inputs = model.prepare_inputs_for_generation(
            current_full_input_ids,
            past_key_values=past,
            attention_mask=current_attention_mask,
            use_cache=True,
        )

        out = model(
            **model_inputs,
            return_dict=True,
        )

        past = out.past_key_values

        next_token = torch.argmax(
            out.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated_tokens.append(int(next_token[0, 0].item()))

        current_full_input_ids = torch.cat(
            [current_full_input_ids, next_token],
            dim=-1,
        )

        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    (current_attention_mask.shape[0], 1),
                    dtype=current_attention_mask.dtype,
                    device=current_attention_mask.device,
                ),
            ],
            dim=-1,
        )

        maybe_nvtx_pop(enable_nvtx)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    maybe_nvtx_pop(enable_nvtx)

    peak_mem = torch.cuda.max_memory_allocated(device)

    return {
        "final_past_key_values": past,
        "decode_steps": int(decode_steps),
        "decode_latency_sec": float(t1 - t0),
        "decode_tok_per_sec": float(decode_steps / (t1 - t0)),
        "decode_peak_cuda_memory_bytes": int(peak_mem),
        "decode_peak_cuda_memory_gb": float(peak_mem / (1024 ** 3)),
        "generated_tokens_count_including_prefill_choice": len(generated_tokens),
        "generated_token_ids_prefix": generated_tokens[:16],
        "generated_token_ids_suffix": generated_tokens[-16:],
    }


def dense_decode_cache_report(past_key_values):
    return {
        "final_dense_cache_seq_len": dense_past_seq_len(past_key_values),
        "final_dense_kv_bytes": dense_past_num_bytes(past_key_values),
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cached_model",
        type=str,
        default="runs/svd_uniform_08/model/svd_uniform_08_cached.pt",
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
        default=128,
    )

    parser.add_argument(
        "--enable_nvtx",
        action="store_true",
        help="Enable NVTX ranges for Nsight Systems profiling.",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# Main benchmark
# ============================================================

def main():
    args = parse_args()

    device = args.device
    cached_model = args.cached_model
    prompt_len = args.prompt_len
    decode_steps = args.decode_steps
    enable_nvtx = args.enable_nvtx

    if args.output_json is None:
        out_path = (
            f"runs/svd_uniform_08/eval/"
            f"bench_turboquant_long_decode_only_p{prompt_len}_d{decode_steps}.json"
        )
    else:
        out_path = args.output_json

    results = {
        "config": {
            "cached_model": cached_model,
            "prompt_len": prompt_len,
            "decode_steps": decode_steps,
            "device": device,
            "enable_nvtx": bool(enable_nvtx),
        }
    }

    # ============================================================
    # A. Baseline: SVD + fp16 KV
    # ============================================================
    print("========== A. Baseline SVD + fp16 KV ==========")

    model_base, tokenizer_base = load_cached_svd_model(
        cached_model,
        device,
    )

    input_ids_base, attention_mask_base = build_fixed_length_prompt(
        tokenizer_base,
        prompt_len=prompt_len,
        device=device,
    )

    prefill_base = prefill_dense_cache(
        model=model_base,
        input_ids=input_ids_base,
        attention_mask=attention_mask_base,
        device=device,
        enable_nvtx=enable_nvtx,
        nvtx_name="baseline_dense_prefill",
    )

    decode_base = manual_decode_loop(
        model=model_base,
        input_ids=input_ids_base,
        attention_mask=attention_mask_base,
        first_token=prefill_base["next_token"],
        past_key_values=prefill_base["past_key_values"],
        decode_steps=decode_steps,
        device=device,
        enable_nvtx=enable_nvtx,
        nvtx_name="baseline_decode_loop",
    )

    dense_final_report = dense_decode_cache_report(
        decode_base["final_past_key_values"]
    )

    baseline_result = {
        "prefill_latency_sec": prefill_base["prefill_latency_sec"],
        "prefill_peak_cuda_memory_bytes": prefill_base["prefill_peak_cuda_memory_bytes"],
        "prefill_peak_cuda_memory_gb": prefill_base["prefill_peak_cuda_memory_gb"],
        "prefill_cache_seq_len": prefill_base["prefill_cache_seq_len"],
        "prefill_dense_kv_bytes": prefill_base["prefill_dense_kv_bytes"],
        "decode_latency_sec": decode_base["decode_latency_sec"],
        "decode_tok_per_sec": decode_base["decode_tok_per_sec"],
        "decode_peak_cuda_memory_bytes": decode_base["decode_peak_cuda_memory_bytes"],
        "decode_peak_cuda_memory_gb": decode_base["decode_peak_cuda_memory_gb"],
        "generated_token_ids_prefix": decode_base["generated_token_ids_prefix"],
        "generated_token_ids_suffix": decode_base["generated_token_ids_suffix"],
        **dense_final_report,
    }

    print(json.dumps(baseline_result, indent=2))
    results["baseline_fp16_kv"] = baseline_result

    del decode_base
    del prefill_base
    del model_base
    torch.cuda.empty_cache()

    # ============================================================
    # B. TurboQuant CUDA runtime cache
    #    Dense prefill -> offline runtime-cache hydration -> CUDA decode
    # ============================================================
    print()
    print("========== B. Packed TurboQuant-K + CUDA decode score ==========")

    model_tq, tokenizer_tq = load_cached_svd_model(
        cached_model,
        device,
    )

    input_ids_tq, attention_mask_tq = build_fixed_length_prompt(
        tokenizer_tq,
        prompt_len=prompt_len,
        device=device,
    )

    # Fast dense prefill before patching.
    prefill_tq_dense = prefill_dense_cache(
        model=model_tq,
        input_ids=input_ids_tq,
        attention_mask=attention_mask_tq,
        device=device,
        enable_nvtx=enable_nvtx,
        nvtx_name="tq_dense_prefill_before_hydration",
    )

    # Patch model only for decode.
    tq_state = patch_llama_model_with_turboquant_scores(
        model_tq,
        qjl_m=256,
        device=device,
        use_cuda_decode_score=True,
    )

    runtime_cache, runtime_cache_hydration_sec = (
        hydrate_turboquant_runtime_cache_from_dense_past(
            dense_past=prefill_tq_dense["past_key_values"],
            tq_state=tq_state,
            num_layers=len(model_tq.model.layers),
            device=device,
            max_cache_len=prompt_len + decode_steps + 1,
            enable_nvtx=enable_nvtx,
        )
    )

    # Dense past is no longer needed once compressed cache is hydrated.
    del prefill_tq_dense["past_key_values"]
    torch.cuda.empty_cache()

    decode_tq = manual_decode_loop(
        model=model_tq,
        input_ids=input_ids_tq,
        attention_mask=attention_mask_tq,
        first_token=prefill_tq_dense["next_token"],
        past_key_values=runtime_cache,
        decode_steps=decode_steps,
        device=device,
        enable_nvtx=enable_nvtx,
        nvtx_name="turboquant_cuda_decode_loop",
    )

    runtime_cache_report = decode_tq[
        "final_past_key_values"
    ].report()

    tq_result = {
        "dense_prefill_latency_sec": prefill_tq_dense["prefill_latency_sec"],
        "dense_prefill_peak_cuda_memory_bytes": prefill_tq_dense["prefill_peak_cuda_memory_bytes"],
        "dense_prefill_peak_cuda_memory_gb": prefill_tq_dense["prefill_peak_cuda_memory_gb"],
        "dense_prefill_cache_seq_len": prefill_tq_dense["prefill_cache_seq_len"],
        "dense_prefill_kv_bytes": prefill_tq_dense["prefill_dense_kv_bytes"],
        "runtime_cache_hydration_sec_excluded_from_decode": runtime_cache_hydration_sec,
        "decode_latency_sec": decode_tq["decode_latency_sec"],
        "decode_tok_per_sec": decode_tq["decode_tok_per_sec"],
        "decode_peak_cuda_memory_bytes": decode_tq["decode_peak_cuda_memory_bytes"],
        "decode_peak_cuda_memory_gb": decode_tq["decode_peak_cuda_memory_gb"],
        "generated_token_ids_prefix": decode_tq["generated_token_ids_prefix"],
        "generated_token_ids_suffix": decode_tq["generated_token_ids_suffix"],
        "runtime_cache_report": runtime_cache_report,
    }

    print(json.dumps(tq_result, indent=2))
    results["turboquant_packed_k_cuda_decode"] = tq_result

    # ============================================================
    # C. Summary
    # ============================================================
    baseline_decode_sec = baseline_result["decode_latency_sec"]
    tq_decode_sec = tq_result["decode_latency_sec"]

    baseline_final_kv_bytes = baseline_result["final_dense_kv_bytes"]

    tq_compressed_k_bytes = runtime_cache_report["compressed_k"]["actual_storage_bytes"]
    tq_fp_v_bytes = runtime_cache_report["fp_value_cache_bytes"]
    tq_total_cache_bytes = tq_compressed_k_bytes + tq_fp_v_bytes

    summary = {
        "decode_latency_speedup_fp16_over_tq": (
            baseline_decode_sec / tq_decode_sec
            if tq_decode_sec > 0 else None
        ),
        "baseline_decode_tok_per_sec": baseline_result["decode_tok_per_sec"],
        "tq_cuda_decode_tok_per_sec": tq_result["decode_tok_per_sec"],
        "baseline_final_dense_kv_bytes": int(baseline_final_kv_bytes),
        "tq_final_compressed_k_bytes": int(tq_compressed_k_bytes),
        "tq_final_fp16_v_bytes": int(tq_fp_v_bytes),
        "tq_final_total_cache_bytes": int(tq_total_cache_bytes),
        "overall_cache_ratio_tq_over_fp16": (
            float(tq_total_cache_bytes) / float(baseline_final_kv_bytes)
        ),
        "overall_cache_reduction_percent": (
            100.0 * (
                1.0 - float(tq_total_cache_bytes) / float(baseline_final_kv_bytes)
            )
        ),
    }

    print()
    print("========== Summary ==========")
    print(json.dumps(summary, indent=2))
    results["summary"] = summary

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"[Save] {out_path}")
    print("[PASS] Long-context decode-only benchmark completed.")


if __name__ == "__main__":
    main()