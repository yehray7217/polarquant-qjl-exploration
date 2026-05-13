from pathlib import Path
import sys
import json

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
def prefill_once(model, tokenizer, prompt: str, runtime_cache, device: str):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=runtime_cache,
        use_cache=True,
        return_dict=True,
    )

    logits = out.logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits": logits,
        "next_token": next_token,
        "cache": out.past_key_values,
    }


@torch.no_grad()
def decode_once(
    model,
    previous_input_ids: torch.Tensor,
    previous_attention_mask: torch.Tensor,
    next_token: torch.Tensor,
    runtime_cache,
):
    full_input_ids = torch.cat([previous_input_ids, next_token], dim=-1)
    full_attention_mask = torch.cat(
        [
            previous_attention_mask,
            torch.ones(
                (previous_attention_mask.shape[0], 1),
                dtype=previous_attention_mask.dtype,
                device=previous_attention_mask.device,
            ),
        ],
        dim=-1,
    )

    model_inputs = model.prepare_inputs_for_generation(
        full_input_ids,
        past_key_values=runtime_cache,
        attention_mask=full_attention_mask,
        use_cache=True,
    )

    out = model(
        **model_inputs,
        return_dict=True,
    )

    logits = out.logits
    next_token_2 = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    return {
        "full_input_ids": full_input_ids,
        "full_attention_mask": full_attention_mask,
        "model_inputs": model_inputs,
        "logits": logits,
        "next_token": next_token_2,
        "cache": out.past_key_values,
    }


@torch.no_grad()
def topk_info(logits_last: torch.Tensor, k: int = 5):
    values, indices = torch.topk(logits_last, k=k, dim=-1)
    return {
        "indices": indices[0].tolist(),
        "values": values[0].tolist(),
        "top1_minus_top2": float((values[0, 0] - values[0, 1]).item()),
    }


@torch.no_grad()
def main():
    device = "cuda:0"
    cached_model = "runs/svd_uniform_08/model/svd_uniform_08_cached.pt"
    prompt = "TurboQuant is useful for language model inference because"

    # ============================================================
    # Python decode score model
    # ============================================================
    model_py, tokenizer_py = load_cached_svd_model(cached_model, device)

    tq_state_py = patch_llama_model_with_turboquant_scores(
        model_py,
        qjl_m=256,
        device=device,
        use_cuda_decode_score=False,
    )

    cache_py = TurboQuantRuntimeCache(
        num_layers=len(model_py.model.layers),
        rotation=tq_state_py.rotation,
        centroids=tq_state_py.centroids,
        sketch=tq_state_py.sketch,
    )

    # ============================================================
    # CUDA decode score model
    # ============================================================
    model_cuda, tokenizer_cuda = load_cached_svd_model(cached_model, device)

    tq_state_cuda = patch_llama_model_with_turboquant_scores(
        model_cuda,
        qjl_m=256,
        device=device,
        use_cuda_decode_score=True,
    )

    cache_cuda = TurboQuantRuntimeCache(
        num_layers=len(model_cuda.model.layers),
        rotation=tq_state_cuda.rotation,
        centroids=tq_state_cuda.centroids,
        sketch=tq_state_cuda.sketch,
    )

    # ============================================================
    # Prefill: both paths should be Python fallback and match
    # ============================================================
    prefill_py = prefill_once(
        model_py,
        tokenizer_py,
        prompt,
        cache_py,
        device,
    )

    prefill_cuda = prefill_once(
        model_cuda,
        tokenizer_cuda,
        prompt,
        cache_cuda,
        device,
    )

    prefill_diff = torch.abs(prefill_py["logits"] - prefill_cuda["logits"])
    prefill_max_abs_diff = prefill_diff.max().item()
    prefill_mean_abs_diff = prefill_diff.mean().item()

    print("========== Prefill parity ==========")
    print(f"prefill max_abs_diff  = {prefill_max_abs_diff:.6e}")
    print(f"prefill mean_abs_diff = {prefill_mean_abs_diff:.6e}")
    print("prefill next token py  =", prefill_py["next_token"][0].tolist())
    print("prefill next token cuda=", prefill_cuda["next_token"][0].tolist())
    print()

    assert torch.equal(
        prefill_py["next_token"],
        prefill_cuda["next_token"],
    ), "Prefill next token mismatch."

    # Use exactly the same first generated token for both decode paths.
    first_generated_token = prefill_py["next_token"]

    # ============================================================
    # Decode step 1:
    # Python packed score vs CUDA packed score
    # ============================================================
    decode_py = decode_once(
        model_py,
        prefill_py["input_ids"],
        prefill_py["attention_mask"],
        first_generated_token,
        prefill_py["cache"],
    )

    decode_cuda = decode_once(
        model_cuda,
        prefill_cuda["input_ids"],
        prefill_cuda["attention_mask"],
        first_generated_token,
        prefill_cuda["cache"],
    )

    logits_py = decode_py["logits"][:, -1, :]
    logits_cuda = decode_cuda["logits"][:, -1, :]

    logit_diff = torch.abs(logits_py - logits_cuda)

    max_abs_diff = logit_diff.max().item()
    mean_abs_diff = logit_diff.mean().item()

    topk_py = topk_info(logits_py, k=5)
    topk_cuda = topk_info(logits_cuda, k=5)

    print("========== Decode-step logits parity ==========")
    print(f"logits max_abs_diff  = {max_abs_diff:.6e}")
    print(f"logits mean_abs_diff = {mean_abs_diff:.6e}")
    print()
    print("Python top-k:", json.dumps(topk_py, indent=2))
    print("CUDA   top-k:", json.dumps(topk_cuda, indent=2))
    print()
    print("decode next token py  =", decode_py["next_token"][0].tolist())
    print("decode next token cuda=", decode_cuda["next_token"][0].tolist())
    print()

    out = {
        "prefill_max_abs_diff": prefill_max_abs_diff,
        "prefill_mean_abs_diff": prefill_mean_abs_diff,
        "decode_logits_max_abs_diff": max_abs_diff,
        "decode_logits_mean_abs_diff": mean_abs_diff,
        "python_topk": topk_py,
        "cuda_topk": topk_cuda,
        "decode_next_token_python": decode_py["next_token"][0].tolist(),
        "decode_next_token_cuda": decode_cuda["next_token"][0].tolist(),
    }

    out_path = "runs/svd_uniform_08/eval/turboquant_cuda_decode_model_parity.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[Save] {out_path}")
    print("[PASS] CUDA decode model parity diagnostic completed.")


if __name__ == "__main__":
    main()
