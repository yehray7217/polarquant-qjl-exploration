from pathlib import Path
import sys
import json
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from turboquant.mse_quant import (
    make_random_rotation,
    get_2bit_centroids,
)
from turboquant.qjl import make_gaussian_sketch
from turboquant.prod_quant import (
    turboquant_prod_quantize_3bit,
    turboquant_prod_inner_product_estimate,
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


def get_layer_k_cache(past_key_values, layer_idx: int):
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values[layer_idx][0]

    key_cache = getattr(past_key_values, "key_cache", None)
    if key_cache is not None:
        return key_cache[layer_idx]

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
        "TurboQuant compresses key-value cache representations to reduce memory "
        "traffic during autoregressive decoding. This script extracts real query "
        "and key tensors from an SVD-compressed LLaMA model and evaluates "
        "TurboQuant_prod across all attention layers. "
    )
    prompt = prompt * 8

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    num_layers = len(model.model.layers)
    captured_hidden_states = {}

    handles = []

    def make_pre_hook(layer_idx: int):
        def pre_hook(module, args, kwargs):
            if len(args) > 0:
                hidden_states = args[0]
            else:
                hidden_states = kwargs["hidden_states"]
            captured_hidden_states[layer_idx] = hidden_states.detach()
        return pre_hook

    for layer_idx in range(num_layers):
        handle = model.model.layers[layer_idx].self_attn.register_forward_pre_hook(
            make_pre_hook(layer_idx),
            with_kwargs=True,
        )
        handles.append(handle)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    for handle in handles:
        handle.remove()

    if len(captured_hidden_states) != num_layers:
        raise RuntimeError(
            f"Captured hidden states for {len(captured_hidden_states)} layers, "
            f"expected {num_layers}."
        )

    past_key_values = outputs.past_key_values

    # Shared TurboQuant config for all layers.
    qjl_m = 256
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    rotation = make_random_rotation(
        d=head_dim,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    centroids = get_2bit_centroids(
        d=head_dim,
        device=device,
        dtype=torch.float32,
    )

    sketch = make_gaussian_sketch(
        d=head_dim,
        m=qjl_m,
        device=device,
        dtype=torch.float32,
        seed=456,
    )

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    all_results = []

    print("========== Real SVD-LLaMA TurboQuant_prod attention sweep ==========")
    print(f"num_layers = {num_layers}")
    print(f"QJL sketch m = {qjl_m}")
    print()

    for layer_idx in range(num_layers):
        attn = model.model.layers[layer_idx].self_attn
        hidden_states = captured_hidden_states[layer_idx]

        key_cache = get_layer_k_cache(past_key_values, layer_idx)

        # key_cache: [B, H_kv, T, D]
        B, H_kv, T, D = key_cache.shape
        H_q = model.config.num_attention_heads
        num_key_value_groups = H_q // H_kv

        if num_key_value_groups > 1:
            key_for_scores = key_cache[:, :, None, :, :].expand(
                B, H_kv, num_key_value_groups, T, D
            ).reshape(B, H_q, T, D)
        else:
            key_for_scores = key_cache

        # Build real Q states for this layer.
        q_proj = attn.q_proj(hidden_states)
        q_states = q_proj.view(B, T, H_q, D).transpose(1, 2)  # [B,H,T,D]

        position_ids = torch.arange(
            T,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)

        # This matches your local transformers version.
        cos, sin = attn.rotary_emb(
            q_states,
            seq_len=T,
        )

        q_states_rope, _ = apply_rotary_pos_emb(
            q_states,
            q_states,
            cos,
            sin,
            position_ids,
        )

        q_last = q_states_rope[:, :, -1:, :]  # [B,H,1,D]

        scale = 1.0 / math.sqrt(D)

        # Reference attention score.
        scores_ref = torch.matmul(
            q_last.float(),
            key_for_scores.float().transpose(-1, -2),
        ) * scale  # [B,H,1,T]

        probs_ref = F.softmax(scores_ref, dim=-1)

        # TurboQuant-prod encoding for all keys in this layer.
        key_flat = key_for_scores.float().reshape(-1, D)

        k_enc = turboquant_prod_quantize_3bit(
            x=key_flat,
            rotation=rotation,
            centroids=centroids,
            sketch=sketch,
        )

        # Repeat each final-token query T times so ordering matches key_flat.
        q_repeat = (
            q_last.float()
            .reshape(B * H_q, 1, D)
            .expand(-1, T, -1)
            .reshape(-1, D)
        )

        est_dot = turboquant_prod_inner_product_estimate(
            q=q_repeat,
            encoding=k_enc,
            rotation=rotation,
            centroids=centroids,
            sketch=sketch,
        )

        scores_tq = est_dot.reshape(B, H_q, 1, T) * scale
        probs_tq = F.softmax(scores_tq, dim=-1)

        score_err = scores_tq - scores_ref
        score_mae = torch.mean(torch.abs(score_err)).item()
        score_rmse = torch.sqrt(torch.mean(score_err ** 2)).item()

        prob_err = probs_tq - probs_ref
        prob_mae = torch.mean(torch.abs(prob_err)).item()
        prob_max = torch.max(torch.abs(prob_err)).item()

        top1_ref = torch.argmax(probs_ref, dim=-1)
        top1_tq = torch.argmax(probs_tq, dim=-1)
        top1_agreement = (top1_ref == top1_tq).float().mean().item()

        result = {
            "layer_idx": layer_idx,
            "B": B,
            "H_q": H_q,
            "H_kv": H_kv,
            "T": T,
            "D": D,
            "score_mae": score_mae,
            "score_rmse": score_rmse,
            "probability_mae": prob_mae,
            "probability_max_error": prob_max,
            "top1_attention_agreement": top1_agreement,
        }
        all_results.append(result)

        print(
            f"L{layer_idx:02d} "
            f"score_mae={score_mae:.6e} "
            f"score_rmse={score_rmse:.6e} "
            f"prob_mae={prob_mae:.6e} "
            f"prob_max={prob_max:.6e} "
            f"top1={top1_agreement:.6f}"
        )

    avg_score_mae = sum(x["score_mae"] for x in all_results) / len(all_results)
    avg_score_rmse = sum(x["score_rmse"] for x in all_results) / len(all_results)
    avg_prob_mae = sum(x["probability_mae"] for x in all_results) / len(all_results)
    avg_prob_max = sum(x["probability_max_error"] for x in all_results) / len(all_results)
    avg_top1 = sum(x["top1_attention_agreement"] for x in all_results) / len(all_results)

    min_top1 = min(x["top1_attention_agreement"] for x in all_results)
    max_prob_mae = max(x["probability_mae"] for x in all_results)

    print()
    print("----- Average over 32 layers -----")
    print(f"avg score MAE                  = {avg_score_mae:.6e}")
    print(f"avg score RMSE                 = {avg_score_rmse:.6e}")
    print(f"avg probability MAE            = {avg_prob_mae:.6e}")
    print(f"avg probability max error      = {avg_prob_max:.6e}")
    print(f"avg top-1 attention agreement  = {avg_top1:.6f}")
    print(f"min top-1 attention agreement  = {min_top1:.6f}")
    print(f"max layer probability MAE      = {max_prob_mae:.6e}")

    out_path = "runs/svd_uniform_08/eval/turboquant_real_attention_all_layers.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "avg_score_mae": avg_score_mae,
                "avg_score_rmse": avg_score_rmse,
                "avg_probability_mae": avg_prob_mae,
                "avg_probability_max_error": avg_prob_max,
                "avg_top1_attention_agreement": avg_top1,
                "min_top1_attention_agreement": min_top1,
                "max_layer_probability_mae": max_prob_mae,
                "layers": all_results,
            },
            f,
            indent=2,
        )

    print(f"[Save] {out_path}")

    # Loose acceptance gates for first all-layer sweep.
    assert avg_prob_mae < 0.002, f"avg probability MAE too high: {avg_prob_mae}"
    assert avg_top1 > 0.40, f"avg top-1 agreement too low: {avg_top1}"

    print("[PASS] Real all-layer TurboQuant_prod attention sweep passed.")


if __name__ == "__main__":
    main()
