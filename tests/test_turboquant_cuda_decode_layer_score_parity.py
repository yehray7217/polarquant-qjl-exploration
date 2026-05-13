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
def main():
    device = "cuda:0"
    cached_model = "runs/svd_uniform_08/model/svd_uniform_08_cached.pt"
    prompt = "TurboQuant is useful for language model inference because"

    model, tokenizer = load_cached_svd_model(
        cached_model,
        device,
    )

    tq_state = patch_llama_model_with_turboquant_scores(
        model,
        qjl_m=256,
        device=device,
        use_cuda_decode_score=False,      # actual forward uses Python score
        compare_cuda_decode_score=True,   # but also computes CUDA score for diagnostics
    )

    runtime_cache = TurboQuantRuntimeCache(
        num_layers=len(model.model.layers),
        rotation=tq_state.rotation,
        centroids=tq_state.centroids,
        sketch=tq_state.sketch,
    )

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # ------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------
    out_prefill = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=runtime_cache,
        use_cache=True,
        return_dict=True,
    )

    first_token = torch.argmax(
        out_prefill.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    print("first generated token:", first_token[0].tolist())

    # ------------------------------------------------------------
    # One decode step
    # ------------------------------------------------------------
    full_input_ids = torch.cat([input_ids, first_token], dim=-1)
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
        past_key_values=out_prefill.past_key_values,
        attention_mask=full_attention_mask,
        use_cache=True,
    )

    _ = model(
        **model_inputs,
        return_dict=True,
    )

    diagnostics = tq_state.decode_score_diagnostics

    print("========== Per-layer Python vs CUDA decode score parity ==========")
    for d in diagnostics:
        print(
            f"L{d['layer_idx']:02d} "
            f"seq={d['seq_len']} "
            f"max_diff={d['max_abs_diff']:.6e} "
            f"mean_diff={d['mean_abs_diff']:.6e} "
            f"py_abs_max={d['python_abs_max']:.6e} "
            f"cuda_abs_max={d['cuda_abs_max']:.6e}"
        )

    max_layer = max(diagnostics, key=lambda x: x["max_abs_diff"])

    print()
    print("----- Worst layer -----")
    print(json.dumps(max_layer, indent=2))

    out_path = "runs/svd_uniform_08/eval/turboquant_cuda_decode_layer_score_parity.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "diagnostics": diagnostics,
                "worst_layer": max_layer,
            },
            f,
            indent=2,
        )

    print(f"[Save] {out_path}")
    print("[PASS] CUDA decode layer-score parity diagnostic completed.")


if __name__ == "__main__":
    main()
