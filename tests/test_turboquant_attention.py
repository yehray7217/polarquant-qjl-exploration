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


@torch.no_grad()
def main():
    torch.manual_seed(0)

    # Llama-2-7B-like attention dimensions.
    batch_size = 2
    num_heads = 32
    query_len = 1
    kv_len = 256
    head_dim = 128

    # TurboQuant-prod config.
    qjl_m = 256

    rotation = make_random_rotation(
        d=head_dim,
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )

    centroids = get_2bit_centroids(
        d=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    sketch = make_gaussian_sketch(
        d=head_dim,
        m=qjl_m,
        device="cpu",
        dtype=torch.float32,
        seed=456,
    )

    # Q: [B, H, 1, D]
    # K: [B, H, T, D]
    # V: [B, H, T, D]
    q = torch.randn(batch_size, num_heads, query_len, head_dim)
    k = torch.randn(batch_size, num_heads, kv_len, head_dim)
    v = torch.randn(batch_size, num_heads, kv_len, head_dim)

    # ------------------------------------------------------------
    # Baseline fp32 attention
    # ------------------------------------------------------------
    scale = 1.0 / math.sqrt(head_dim)

    # [B, H, 1, T]
    scores_ref = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs_ref = F.softmax(scores_ref, dim=-1)
    out_ref = torch.matmul(probs_ref, v)  # [B, H, 1, D]

    # ------------------------------------------------------------
    # TurboQuant-prod K encoding
    # Flatten K to [B*H*T, D], quantize, then reshape estimates back.
    # ------------------------------------------------------------
    k_flat = k.reshape(-1, head_dim)

    k_enc = turboquant_prod_quantize_3bit(
        x=k_flat,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    # For each query vector, estimate dot against all K vectors
    # within the same [B, H] group.
    q_flat = q.reshape(batch_size * num_heads, query_len, head_dim)
    k_enc_rows_per_group = kv_len

    tq_scores_groups = []

    for group_idx in range(batch_size * num_heads):
        q_group = q_flat[group_idx]  # [1, D]

        start = group_idx * k_enc_rows_per_group
        end = start + k_enc_rows_per_group

        # Slice encoding for the K rows belonging to this head.
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

        # Broadcast query [1, D] -> [T, D]
        q_repeated = q_group.expand(kv_len, -1)

        est_dot = turboquant_prod_inner_product_estimate(
            q=q_repeated,
            encoding=sliced_enc,
            rotation=rotation,
            centroids=centroids,
            sketch=sketch,
        )  # [T]

        tq_scores_groups.append(est_dot)

    # [B*H, T] -> [B, H, 1, T]
    scores_tq = torch.stack(tq_scores_groups, dim=0)
    scores_tq = scores_tq.reshape(batch_size, num_heads, query_len, kv_len)
    scores_tq = scores_tq * scale

    probs_tq = F.softmax(scores_tq, dim=-1)
    out_tq = torch.matmul(probs_tq, v)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    score_abs_err = torch.abs(scores_tq - scores_ref)
    score_mae = score_abs_err.mean().item()
    score_rmse = torch.sqrt(torch.mean((scores_tq - scores_ref) ** 2)).item()

    probs_abs_err = torch.abs(probs_tq - probs_ref)
    probs_mae = probs_abs_err.mean().item()
    probs_max_err = probs_abs_err.max().item()

    out_abs_err = torch.abs(out_tq - out_ref)
    out_mae = out_abs_err.mean().item()
    out_rmse = torch.sqrt(torch.mean((out_tq - out_ref) ** 2)).item()

    out_cos = F.cosine_similarity(
        out_ref.reshape(-1, head_dim),
        out_tq.reshape(-1, head_dim),
        dim=-1,
    ).mean().item()

    top1_ref = torch.argmax(probs_ref, dim=-1)
    top1_tq = torch.argmax(probs_tq, dim=-1)
    top1_agreement = (top1_ref == top1_tq).float().mean().item()

    print("========== TurboQuant attention correctness test ==========")
    print(f"batch_size                    = {batch_size}")
    print(f"num_heads                     = {num_heads}")
    print(f"query_len                     = {query_len}")
    print(f"kv_len                        = {kv_len}")
    print(f"head_dim                      = {head_dim}")
    print(f"QJL sketch m                  = {qjl_m}")
    print()
    print("----- Attention score error -----")
    print(f"score MAE                     = {score_mae:.6e}")
    print(f"score RMSE                    = {score_rmse:.6e}")
    print()
    print("----- Softmax probability error -----")
    print(f"probability MAE               = {probs_mae:.6e}")
    print(f"probability max error         = {probs_max_err:.6e}")
    print(f"top-1 attention agreement     = {top1_agreement:.6f}")
    print()
    print("----- Attention output error -----")
    print(f"output MAE                    = {out_mae:.6e}")
    print(f"output RMSE                   = {out_rmse:.6e}")
    print(f"output cosine similarity      = {out_cos:.6f}")

    # Sanity thresholds.
    # These are intentionally loose for the first correctness gate.
    assert out_cos > 0.90, f"attention output cosine too low: {out_cos}"
    assert top1_agreement > 0.50, f"top-1 attention agreement too low: {top1_agreement}"

    print("[PASS] TurboQuant attention correctness sanity check passed.")


if __name__ == "__main__":
    main()
