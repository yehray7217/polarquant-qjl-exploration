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


def register_decoder_layer_hooks(model, store: dict):
    handles = []

    for layer_idx, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook(module, args, output):
                # LlamaDecoderLayer forward output:
                # output[0] = hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                store[idx] = hidden.detach().float().cpu()
            return hook

        handles.append(layer.register_forward_hook(make_hook(layer_idx)))

    return handles


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

    next_token = torch.argmax(
        out.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "next_token": next_token,
        "cache": out.past_key_values,
        "logits": out.logits,
    }


@torch.no_grad()
def decode_once(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    next_token: torch.Tensor,
    runtime_cache,
):
    full_input_ids = torch.cat([input_ids, next_token], dim=-1)

    full_attention_mask = torch.cat(
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

    return out


@torch.no_grad()
def main():
    device = "cuda:0"
    cached_model = "runs/svd_uniform_08/model/svd_uniform_08_cached.pt"
    prompt = "TurboQuant is useful for language model inference because"

    # ============================================================
    # Python-score model
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
    # CUDA-score model
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
    # Prefill
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

    assert torch.equal(
        prefill_py["next_token"],
        prefill_cuda["next_token"],
    ), "Prefill next token mismatch."

    first_token = prefill_py["next_token"]

    # ============================================================
    # Register layer hooks only for decode step
    # ============================================================
    hidden_py = {}
    hidden_cuda = {}

    handles_py = register_decoder_layer_hooks(model_py, hidden_py)
    handles_cuda = register_decoder_layer_hooks(model_cuda, hidden_cuda)

    out_py = decode_once(
        model_py,
        prefill_py["input_ids"],
        prefill_py["attention_mask"],
        first_token,
        prefill_py["cache"],
    )

    out_cuda = decode_once(
        model_cuda,
        prefill_cuda["input_ids"],
        prefill_cuda["attention_mask"],
        first_token,
        prefill_cuda["cache"],
    )

    for h in handles_py + handles_cuda:
        h.remove()

    # ============================================================
    # Per-layer hidden drift
    # ============================================================
    diagnostics = []

    print("========== Decode hidden-state drift: Python score vs CUDA score ==========")

    for layer_idx in range(len(model_py.model.layers)):
        x_py = hidden_py[layer_idx]
        x_cuda = hidden_cuda[layer_idx]

        diff = torch.abs(x_py - x_cuda)

        rec = {
            "layer_idx": layer_idx,
            "hidden_shape": list(x_py.shape),
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            "python_abs_max": float(x_py.abs().max().item()),
            "cuda_abs_max": float(x_cuda.abs().max().item()),
        }
        diagnostics.append(rec)

        print(
            f"L{layer_idx:02d} "
            f"max_diff={rec['max_abs_diff']:.6e} "
            f"mean_diff={rec['mean_abs_diff']:.6e} "
            f"py_abs_max={rec['python_abs_max']:.6e} "
            f"cuda_abs_max={rec['cuda_abs_max']:.6e}"
        )

    # ============================================================
    # Final logits drift
    # ============================================================
    logits_py = out_py.logits[:, -1, :].float().cpu()
    logits_cuda = out_cuda.logits[:, -1, :].float().cpu()

    logit_diff = torch.abs(logits_py - logits_cuda)

    print()
    print("========== Final logits drift ==========")
    print(f"logits max_abs_diff  = {logit_diff.max().item():.6e}")
    print(f"logits mean_abs_diff = {logit_diff.mean().item():.6e}")

    out = {
        "layer_hidden_drift": diagnostics,
        "logits_max_abs_diff": float(logit_diff.max().item()),
        "logits_mean_abs_diff": float(logit_diff.mean().item()),
    }

    out_path = "runs/svd_uniform_08/eval/turboquant_cuda_decode_hidden_drift.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[Save] {out_path}")
    print("[PASS] CUDA decode hidden-drift diagnostic completed.")


if __name__ == "__main__":
    main()
