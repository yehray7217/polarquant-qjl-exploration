from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.llama_score_patch import (
    patch_llama_model_with_turboquant_scores,
)
from turboquant.runtime_cache import TurboQuantRuntimeCache


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


@torch.no_grad()
def run_generate_benchmark(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    past_key_values=None,
    n_warmup: int = 1,
    n_runs: int = 3,
):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    prompt_len = input_ids.shape[1]

    # Warmup
    for _ in range(n_warmup):
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()

    latencies = []
    outputs = []

    for _ in range(n_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        t0 = time.perf_counter()

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append(t1 - t0)
        outputs.append(out)

    peak_mem = torch.cuda.max_memory_allocated(device)

    avg_latency = sum(latencies) / len(latencies)
    generated_tokens = max_new_tokens
    decode_tok_per_sec = generated_tokens / avg_latency

    return {
        "prompt_len": int(prompt_len),
        "max_new_tokens": int(max_new_tokens),
        "n_runs": int(n_runs),
        "latencies_sec": latencies,
        "avg_latency_sec": float(avg_latency),
        "approx_generated_tok_per_sec": float(decode_tok_per_sec),
        "peak_cuda_memory_bytes": int(peak_mem),
        "peak_cuda_memory_gb": float(peak_mem / (1024 ** 3)),
        "last_output_token_ids": outputs[-1][0].tolist(),
        "last_output_text": tokenizer.decode(
            outputs[-1][0],
            skip_special_tokens=True,
        ),
    }


def main():
    device = "cuda:0"
    cached_model = "runs/svd_uniform_08/model/svd_uniform_08_cached.pt"

    prompt = "TurboQuant is useful for language model inference because"
    max_new_tokens = 4

    results = {}

    # ============================================================
    # A. Baseline SVD + default fp16 KV cache
    # ============================================================
    model_base, tokenizer_base = load_cached_svd_model(
        cached_model,
        device,
    )

    print("========== Benchmark: baseline SVD fp16 KV ==========")

    baseline_result = run_generate_benchmark(
        model=model_base,
        tokenizer=tokenizer_base,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        past_key_values=None,
        n_warmup=1,
        n_runs=3,
    )

    print(json.dumps(baseline_result, indent=2))
    results["baseline_fp16_kv"] = baseline_result

    del model_base
    torch.cuda.empty_cache()

    # ============================================================
    # B. SVD + packed TurboQuant K runtime cache + fp16 V
    # ============================================================
    model_tq, tokenizer_tq = load_cached_svd_model(
        cached_model,
        device,
    )

    tq_state = patch_llama_model_with_turboquant_scores(
        model_tq,
        qjl_m=256,
        device=device,
    )

    # Build a fresh cache for each benchmark call.
    # For now benchmark a single measured cache instance.
    runtime_cache = TurboQuantRuntimeCache(
        num_layers=len(model_tq.model.layers),
        rotation=tq_state.rotation,
        centroids=tq_state.centroids,
        sketch=tq_state.sketch,
    )

    print()
    print("========== Benchmark: packed TurboQuant K runtime cache ==========")

    tq_result = run_generate_benchmark(
        model=model_tq,
        tokenizer=tokenizer_tq,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        past_key_values=runtime_cache,
        n_warmup=0,
        n_runs=1,
    )

    tq_result["runtime_cache_report"] = runtime_cache.report()

    print(json.dumps(tq_result, indent=2))
    results["turboquant_packed_k_runtime_cache"] = tq_result

    # ============================================================
    # Save results
    # ============================================================
    out_path = "runs/svd_uniform_08/eval/bench_turboquant_runtime_cache_smoke.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"[Save] {out_path}")
    print("[PASS] Runtime cache benchmark smoke completed.")


if __name__ == "__main__":
    main()
