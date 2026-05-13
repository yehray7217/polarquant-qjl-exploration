from pathlib import Path
import sys
import copy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.llama_score_patch import (
    patch_llama_model_with_turboquant_scores,
)


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
def generate_once(model, tokenizer, prompt: str, device: str):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    return out


@torch.no_grad()
def main():
    device = "cuda:0"
    cached_model = "runs/svd_uniform_08/model/svd_uniform_08_cached.pt"

    prompt = (
        "TurboQuant is useful for language model inference because"
    )

    # Baseline SVD model.
    model_ref, tokenizer = load_cached_svd_model(
        cached_model,
        device,
    )

    out_ref = generate_once(
        model_ref,
        tokenizer,
        prompt,
        device,
    )

    text_ref = tokenizer.decode(
        out_ref[0],
        skip_special_tokens=True,
    )

    print("========== Baseline SVD generate ==========")
    print(text_ref)
    print()

    del model_ref
    torch.cuda.empty_cache()

    # TurboQuant score-patched SVD model.
    model_tq, tokenizer_tq = load_cached_svd_model(
        cached_model,
        device,
    )

    patch_llama_model_with_turboquant_scores(
        model_tq,
        qjl_m=256,
        device=device,
    )

    out_tq = generate_once(
        model_tq,
        tokenizer_tq,
        prompt,
        device,
    )

    text_tq = tokenizer_tq.decode(
        out_tq[0],
        skip_special_tokens=True,
    )

    print("========== TurboQuant score-patched SVD generate ==========")
    print(text_tq)
    print()

    print("----- Token IDs -----")
    print("baseline:", out_ref[0].tolist())
    print("tq      :", out_tq[0].tolist())

    print("[PASS] TurboQuant score-path generation smoke test completed.")


if __name__ == "__main__":
    main()
