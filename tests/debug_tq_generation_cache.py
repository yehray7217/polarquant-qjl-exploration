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


def describe_cache(past_key_values, tag: str):
    print(f"\n========== {tag} ==========")
    print("type:", type(past_key_values))

    if past_key_values is None:
        print("cache: None")
        return

    # Legacy tuple/list cache
    if isinstance(past_key_values, (tuple, list)):
        print("legacy cache layers:", len(past_key_values))
        for i in [0, len(past_key_values) - 1]:
            k, v = past_key_values[i]
            print(
                f"L{i:02d}: "
                f"K={tuple(k.shape)}, V={tuple(v.shape)}, "
                f"K_seq_len={k.shape[-2]}"
            )
        return

    # Cache object
    if hasattr(past_key_values, "get_seq_length"):
        print("cache object get_seq_length(0):", past_key_values.get_seq_length(0))
        if hasattr(past_key_values, "seen_tokens"):
            print("cache object seen_tokens:", past_key_values.seen_tokens)

    if hasattr(past_key_values, "key_cache"):
        print("cache object layers:", len(past_key_values.key_cache))
        for i in [0, len(past_key_values.key_cache) - 1]:
            k = past_key_values.key_cache[i]
            v = past_key_values.value_cache[i]
            print(
                f"L{i:02d}: "
                f"K={tuple(k.shape)}, V={tuple(v.shape)}, "
                f"K_seq_len={k.shape[-2]}"
            )


@torch.no_grad()
def main():
    device = "cuda:0"
    cached_model = "runs/svd_uniform_08/model/svd_uniform_08_cached.pt"

    prompt = "TurboQuant is useful for language model inference because"

    model, tokenizer = load_cached_svd_model(
        cached_model,
        device,
    )

    patch_llama_model_with_turboquant_scores(
        model,
        qjl_m=256,
        device=device,
    )

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    print("========== Initial inputs ==========")
    print("prompt:", prompt)
    print("input_ids.shape:", tuple(input_ids.shape))
    print("attention_mask.shape:", tuple(attention_mask.shape))
    print("input token ids:", input_ids[0].tolist())

    # ------------------------------------------------------------
    # Step 1: manual first forward, equivalent to prefill
    # ------------------------------------------------------------
    out1 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    logits1 = out1.logits
    past1 = out1.past_key_values

    print("\n========== After first forward ==========")
    print("logits1.shape:", tuple(logits1.shape))
    describe_cache(past1, "past1 returned by patched model")

    next_token = torch.argmax(logits1[:, -1, :], dim=-1, keepdim=True)
    print("next_token:", next_token[0].tolist())

    # This mirrors generate(): append next token to attention mask.
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

    print("\n========== Inputs before prepare_inputs_for_generation ==========")
    print("full_input_ids.shape:", tuple(full_input_ids.shape))
    print("full_attention_mask.shape:", tuple(full_attention_mask.shape))

    # ------------------------------------------------------------
    # Step 2: let HF generation helper slice the next-step inputs
    # ------------------------------------------------------------
    model_inputs2 = model.prepare_inputs_for_generation(
        full_input_ids,
        past_key_values=past1,
        attention_mask=full_attention_mask,
        use_cache=True,
    )

    print("\n========== prepare_inputs_for_generation output ==========")
    for k, v in model_inputs2.items():
        if torch.is_tensor(v):
            print(f"{k}.shape:", tuple(v.shape))
        else:
            print(f"{k}:", type(v))

    describe_cache(
        model_inputs2["past_key_values"],
        "past cache passed into second forward",
    )

    # ------------------------------------------------------------
    # Step 3: second forward; this should reproduce current crash
    # ------------------------------------------------------------
    print("\n========== Running second forward ==========")
    out2 = model(
        **model_inputs2,
        return_dict=True,
    )

    print("second forward succeeded.")
    print("logits2.shape:", tuple(out2.logits.shape))
    describe_cache(out2.past_key_values, "past2 returned by patched model")


if __name__ == "__main__":
    main()
