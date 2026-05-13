from pathlib import Path
import sys
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
        "and key tensors from an SVD-compressed LLaMA model. "
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

    layer0_attn = model.model.layers[0].self_attn

    captured = {}

    # ------------------------------------------------------------
    # 1. Capture the input hidden states entering layer-0 attention
    # ------------------------------------------------------------
    def pre_hook(module, args, kwargs):
        # Depending on transformers version, hidden_states may be:
        # args[0] or kwargs["hidden_states"]
        if len(args) > 0:
            hidden_states = args[0]
        else:
            hidden_states = kwargs["hidden_states"]
        captured["hidden_states"] = hidden_states.detach()

    handle = layer0_attn.register_forward_pre_hook(
        pre_hook,
        with_kwargs=True,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    handle.remove()

    if "hidden_states" not in captured:
        raise RuntimeError("Failed to capture attention hidden states.")

    hidden_states = captured["hidden_states"]  # [B, T, hidden_size]
    past_key_values = outputs.past_key_values

    # ------------------------------------------------------------
    # 2. Extract real post-RoPE K cache from layer 0
    # ------------------------------------------------------------
    if isinstance(past_key_values, (tuple, list)):
        key_cache_l0 = past_key_values[0][0]
    else:
        key_cache_l0 = past_key_values.key_cache[0]

    # key_cache_l0: [B, H_kv, T, D]
    B, H_kv, T, D = key_cache_l0.shape

    # For Llama-2-7B, H_kv == H_q == 32.
    # Keep explicit code in case model config differs.
    H_q = model.config.num_attention_heads
    num_key_value_groups = H_q // H_kv

    # Repeat K heads if needed to match query heads.
    if num_key_value_groups > 1:
        key_for_scores = key_cache_l0[:, :, None, :, :].expand(
            B, H_kv, num_key_value_groups, T, D
        ).reshape(B, H_q, T, D)
    else:
        key_for_scores = key_cache_l0

    # ------------------------------------------------------------
    # 3. Reconstruct real layer-0 Q for the final prompt token
    # ------------------------------------------------------------
    # hidden_states: [B, T, hidden_size]
    q_proj = layer0_attn.q_proj(hidden_states)

    head_dim = D
    q_states = q_proj.view(B, T, H_q, head_dim).transpose(1, 2)  # [B, H, T, D]

    # Need the final-token query.
    q_last_pre_rope = q_states[:, :, -1:, :]  # [B, H, 1, D]

    # ------------------------------------------------------------
    # 4. Apply RoPE to the query.
    #    We infer RoPE from the attention module's rotary embedding.
    # ------------------------------------------------------------
    seq_len = T
    position_ids = torch.arange(
        seq_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0).expand(B, -1)

    # Different transformers versions expose rotary_emb slightly differently.
    # This path matches common Llama implementations.
    cos, sin = layer0_attn.rotary_emb(
        q_states,
        seq_len=T,
    )

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    q_states_rope, _ = apply_rotary_pos_emb(
        q_states,
        q_states,
        cos,
        sin,
        position_ids,
    )

    q_last = q_states_rope[:, :, -1:, :]  # [B, H, 1, D]

    # ------------------------------------------------------------
    # 5. Baseline real attention score against post-RoPE K cache
    # ------------------------------------------------------------
    scale = 1.0 / math.sqrt(head_dim)

    scores_ref = torch.matmul(
        q_last.float(),
        key_for_scores.float().transpose(-1, -2),
    ) * scale  # [B, H, 1, T]

    probs_ref = F.softmax(scores_ref, dim=-1)

    # ------------------------------------------------------------
    # 6. TurboQuant-prod score estimate for K cache
    # ------------------------------------------------------------
    qjl_m = 256

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

    key_flat = key_for_scores.float().reshape(-1, head_dim)

    k_enc = turboquant_prod_quantize_3bit(
        x=key_flat,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    tq_score_groups = []

    q_grouped = q_last.float().reshape(B * H_q, 1, head_dim)
    rows_per_group = T

    for group_idx in range(B * H_q):
        start = group_idx * rows_per_group
        end = start + rows_per_group

        sliced_enc = type(k_enc)(
            mse=type(k_enc.mse)(
                indices=k_enc.mse.indices[start:end],
                norms=k_enc.mse.norms[start:end],
            ),
            qjl_residual=type(k_enc.qjl_residual)(
                sign_bits=k_enc.qjl_residual.sign_bits[start:end],
                norms=k_enc.qjl_residual.norms[start:end],
            ),
        )

        q_repeated = q_grouped[group_idx].expand(T, -1)

        est_dot = turboquant_prod_inner_product_estimate(
            q=q_repeated,
            encoding=sliced_enc,
            rotation=rotation,
            centroids=centroids,
            sketch=sketch,
        )

        tq_score_groups.append(est_dot)

    scores_tq = torch.stack(tq_score_groups, dim=0)
    scores_tq = scores_tq.reshape(B, H_q, 1, T) * scale

    probs_tq = F.softmax(scores_tq, dim=-1)

    # ------------------------------------------------------------
    # 7. Compare score / softmax metrics
    # ------------------------------------------------------------
    score_err = scores_tq - scores_ref
    score_mae = torch.mean(torch.abs(score_err)).item()
    score_rmse = torch.sqrt(torch.mean(score_err ** 2)).item()

    prob_err = probs_tq - probs_ref
    prob_mae = torch.mean(torch.abs(prob_err)).item()
    prob_max = torch.max(torch.abs(prob_err)).item()

    top1_ref = torch.argmax(probs_ref, dim=-1)
    top1_tq = torch.argmax(probs_tq, dim=-1)
    top1_agreement = (top1_ref == top1_tq).float().mean().item()

    print("========== Real SVD-LLaMA Layer-0 TurboQuant_prod attention test ==========")
    print(f"B                              = {B}")
    print(f"H_q                            = {H_q}")
    print(f"H_kv                           = {H_kv}")
    print(f"T                              = {T}")
    print(f"D                              = {D}")
    print(f"QJL sketch m                   = {qjl_m}")
    print()
    print("----- Real attention score error -----")
    print(f"score MAE                      = {score_mae:.6e}")
    print(f"score RMSE                     = {score_rmse:.6e}")
    print()
    print("----- Real softmax distribution error -----")
    print(f"probability MAE                = {prob_mae:.6e}")
    print(f"probability max error          = {prob_max:.6e}")
    print(f"top-1 attention agreement      = {top1_agreement:.6f}")

    assert prob_mae < 0.01, f"probability MAE too high: {prob_mae}"
    assert top1_agreement > 0.40, f"top-1 agreement too low: {top1_agreement}"

    print("[PASS] Real layer-0 TurboQuant_prod attention sanity check passed.")


if __name__ == "__main__":
    main()
