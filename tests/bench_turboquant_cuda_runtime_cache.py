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
def run_generate_once(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    past_key_values=None,
):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

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

    peak_mem = torch.cuda.max_memory_allocated(device)

    return {
        "prompt_len": int(input_ids.shape[1]),
        "max_new_tokens": int(max_new_tokens),
        "latency_sec": float(t1 - t0),
        "approx_generated_tok_per_sec": float(max_new_tokens / (t1 - t0)),
        "peak_cuda_memory_bytes": int(peak_mem),
        "peak_cuda_memory_gb": float(peak_mem / (1024 ** 3)),
        "output_token_ids": out[0].tolist(),
        "output_text": tokenizer.decode(out[0], skip_special_tokens=True),
    }


def benchmark_turboquant_variant(
    cached_model: str,
    prompt: str,
    max_new_tokens: int,
    device: str,
    use_cuda_decode_score: bool,
):
    model, tokenizer = load_cached_svd_model(
        cached_model,
        device,
    )

    tq_state = patch_llama_model_with_turboquant_scores(
        model,
        qjl_m=256,
        device=device,
        use_cuda_decode_score=use_cuda_decode_score,
    )

    runtime_cache = TurboQuantRuntimeCache(
        num_layers=len(model.model.layers),
        rotation=tq_state.rotation,
        centroids=tq_state.centroids,
        sketch=tq_state.sketch,
    )

    result = run_generate_once(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        past_key_values=runtime_cache,
    )

    result["runtime_cache_report"] = runtime_cache.report()
    result["use_cuda_decode_score"] = bool(use_cuda_decode_score)

    return result


def main():
    device = "cuda:0"
    cached_model = "runs/svd_uniform_08/model/svd_uniform_08_cached.pt"

    prompt = "TurboQuant is useful for language model inference because"
    max_new_tokens = 16

    results = {}

    # ============================================================
    # A. Packed TurboQuant K + Python decode score
    # ============================================================
    print("========== Packed TurboQuant K + Python decode score ==========")

    py_result = benchmark_turboquant_variant(
        cached_model=cached_model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        use_cuda_decode_score=False,
    )

    print(json.dumps(py_result, indent=2))
    results["turboquant_python_decode_score"] = py_result

    torch.cuda.empty_cache()

    # ============================================================
    # B. Packed TurboQuant K + CUDA decode score
    # ============================================================
    print()
    print("========== Packed TurboQuant K + CUDA decode score ==========")

    cuda_result = benchmark_turboquant_variant(
        cached_model=cached_model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        use_cuda_decode_score=True,
    )

    print(json.dumps(cuda_result, indent=2))
    results["turboquant_cuda_decode_score"] = cuda_result

    # ============================================================
    # Derived comparison
    # ============================================================
    speedup_cuda_vs_python = (
        py_result["latency_sec"] / cuda_result["latency_sec"]
        if cuda_result["latency_sec"] > 0 else None
    )

    results["summary"] = {
        "cuda_vs_python_latency_speedup": speedup_cuda_vs_python,
        "python_latency_sec": py_result["latency_sec"],
        "cuda_latency_sec": cuda_result["latency_sec"],
        "python_tok_per_sec": py_result["approx_generated_tok_per_sec"],
        "cuda_tok_per_sec": cuda_result["approx_generated_tok_per_sec"],
    }

    print()
    print("========== Summary ==========")
    print(json.dumps(results["summary"], indent=2))

    out_path = "runs/svd_uniform_08/eval/bench_turboquant_cuda_runtime_cache.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"[Save] {out_path}")
    print("[PASS] CUDA runtime-cache benchmark completed.")


if __name__ == "__main__":
    main()
