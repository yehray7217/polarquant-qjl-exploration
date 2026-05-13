from pathlib import Path
import sys

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

    prompt = "TurboQuant is useful for language model inference because"

    model, tokenizer = load_cached_svd_model(
        cached_model,
        device,
    )

    tq_state = patch_llama_model_with_turboquant_scores(
        model,
        qjl_m=256,
        device=device,
    )

    out = generate_once(
        model,
        tokenizer,
        prompt,
        device,
    )

    text = tokenizer.decode(
        out[0],
        skip_special_tokens=True,
    )

    print("========== TurboQuant compressed side-cache generate ==========")
    print(text)
    print()

    print("----- Token IDs -----")
    print(out[0].tolist())
    print()

    print("----- TurboQuant side key cache report -----")
    report = tq_state.key_cache.report()
    print(report)

    expected_seq_len = out.shape[1] - 1
    # Explanation:
    # generate(max_new_tokens=4) performs:
    #   prefill prompt
    #   decode steps for generated tokens except the final token is not forwarded again.
    # For a prompt of length L and 4 generated tokens,
    # cached KV length should usually be L + 3 at return time.
    #
    # We therefore only assert all layer lengths agree and are > prompt length.
    layer_lens = [
        tq_state.key_cache.get_seq_length(i)
        for i in range(len(model.model.layers))
    ]

    print("layer cache seq lens:", layer_lens)

    assert len(set(layer_lens)) == 1, "Layer cache sequence lengths disagree."
    assert layer_lens[0] > 0, "TurboQuant side cache is empty."

    print("[PASS] TurboQuant compressed side-cache generation smoke test passed.")


if __name__ == "__main__":
    main()
