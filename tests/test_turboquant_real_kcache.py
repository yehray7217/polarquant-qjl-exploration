from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from turboquant.mse_quant import (
    make_random_rotation,
    get_2bit_centroids,
    turboquant_mse_quantize_2bit,
    turboquant_mse_dequantize_2bit,
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


def extract_kv_cache(past_key_values):
    """
    Return list of (key, value), compatible with:
    1. legacy tuple/list past_key_values
    2. HF Cache objects exposing key_cache/value_cache
    """
    if isinstance(past_key_values, (tuple, list)):
        return [(layer[0], layer[1]) for layer in past_key_values]

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)

    if key_cache is not None and value_cache is not None:
        return list(zip(key_cache, value_cache))

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


@torch.no_grad()
def main():
    device = "cuda:0"

    cached_model = "runs/svd_uniform_08/model/svd_uniform_08_cached.pt"

    model, tokenizer = load_cached_svd_model(
        path=cached_model,
        device=device,
    )

    prompt = (
        "TurboQuant compresses KV cache representations for faster long-context "
        "language model inference. This test extracts real cached key vectors from "
        "an SVD-compressed LLaMA model and evaluates reconstruction quality. "
    )

    # Build a moderately sized sequence.
    repeated_prompt = prompt * 16
    enc = tokenizer(
        repeated_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    past = outputs.past_key_values
    kv_layers = extract_kv_cache(past)

    print("========== Real SVD-LLaMA K-cache TurboQuant_mse test ==========")
    print(f"num_layers = {len(kv_layers)}")

    layer_results = []

    for layer_idx, (key, value) in enumerate(kv_layers):
        # key expected shape: [B, H, T, D]
        if key.ndim != 4:
            raise ValueError(
                f"Expected key cache to be 4D [B,H,T,D], got shape={tuple(key.shape)}"
            )

        B, H, T, D = key.shape

        rotation = make_random_rotation(
            d=D,
            device=key.device,
            dtype=torch.float32,
            seed=123,
        )

        centroids = get_2bit_centroids(
            d=D,
            device=key.device,
            dtype=torch.float32,
        )

        key_flat = key.float().reshape(-1, D)

        enc_k = turboquant_mse_quantize_2bit(
            x=key_flat,
            rotation=rotation,
            centroids=centroids,
        )

        key_hat_flat = turboquant_mse_dequantize_2bit(
            encoding=enc_k,
            rotation=rotation,
            centroids=centroids,
        )

        residual = key_flat - key_hat_flat

        sq_err = torch.sum(residual ** 2, dim=-1)
        sq_norm = torch.sum(key_flat ** 2, dim=-1)
        relative_mse = torch.mean(
            sq_err / torch.clamp(sq_norm, min=1e-12)
        ).item()

        residual_norm_ratio = torch.mean(
            torch.linalg.vector_norm(residual, dim=-1)
            / torch.clamp(torch.linalg.vector_norm(key_flat, dim=-1), min=1e-12)
        ).item()

        cosine = F.cosine_similarity(
            key_flat,
            key_hat_flat,
            dim=-1,
        ).mean().item()

        result = {
            "layer_idx": layer_idx,
            "key_shape": [B, H, T, D],
            "relative_mse": relative_mse,
            "residual_norm_ratio": residual_norm_ratio,
            "cosine_similarity": cosine,
        }
        layer_results.append(result)

        print(
            f"L{layer_idx:02d} "
            f"shape={list(key.shape)} "
            f"rel_mse={relative_mse:.6f} "
            f"res_norm_ratio={residual_norm_ratio:.6f} "
            f"cos={cosine:.6f}"
        )

    avg_relative_mse = sum(x["relative_mse"] for x in layer_results) / len(layer_results)
    avg_residual_norm_ratio = sum(x["residual_norm_ratio"] for x in layer_results) / len(layer_results)
    avg_cosine = sum(x["cosine_similarity"] for x in layer_results) / len(layer_results)

    print()
    print("----- Average over layers -----")
    print(f"avg relative MSE            = {avg_relative_mse:.6f}")
    print(f"avg residual norm ratio     = {avg_residual_norm_ratio:.6f}")
    print(f"avg cosine similarity       = {avg_cosine:.6f}")

    out_path = "runs/svd_uniform_08/eval/turboquant_real_kcache_mse.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "avg_relative_mse": avg_relative_mse,
                "avg_residual_norm_ratio": avg_residual_norm_ratio,
                "avg_cosine_similarity": avg_cosine,
                "layers": layer_results,
            },
            f,
            indent=2,
        )

    print(f"[Save] {out_path}")

    # Loose sanity thresholds.
    # Random-vector test gave rel_mse≈0.1155, residual≈0.339, cosine≈0.941.
    # Real K-cache can differ, but should remain in a reasonable regime.
    assert avg_relative_mse < 0.20, f"avg relative MSE too high: {avg_relative_mse}"
    assert avg_cosine > 0.90, f"avg cosine similarity too low: {avg_cosine}"

    print("[PASS] Real K-cache TurboQuant_mse sanity check passed.")


if __name__ == "__main__":
    main()
