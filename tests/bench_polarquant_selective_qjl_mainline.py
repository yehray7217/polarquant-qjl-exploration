#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from turboquant.polarquant import recursive_polar_encode
from turboquant.polarquant_quant import (
    DEFAULT_POLAR_BITS_BY_LEVEL,
    fit_polar_angle_codebooks_from_encodings,
)
from turboquant.polar_prod import turboquant_polar_prod_quantize
from turboquant.qjl import make_gaussian_sketch
from turboquant.packed_meta import build_turboquant_packed_meta_blob
from turboquant.polar_tree_lut import build_tree_l2_factor_lut
from turboquant.qjl_sign_layout import build_qjl_lane_nibble_signs
from turboquant.selective_qjl_pipeline import (
    selective_qjl_sparse_topk_m128_cuda,
    selective_qjl_dense_logits_topk_m128_cuda,
)
from turboquant.turboquant_qjl_norm_early_load_cuda import (
    turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda,
)


def ensure_cuda(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"Expected CUDA device, got {device!r}.")
    return dev


def sync() -> None:
    torch.cuda.synchronize()


@torch.no_grad()
def bench_cuda_ms(
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, Any]:
    out = None
    for _ in range(warmup):
        out = fn()

    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    sync()

    return float(start.elapsed_time(end) / iters), out


@torch.no_grad()
def score_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    topk: int,
) -> dict[str, float]:
    cand = candidate.to(torch.float32)
    ref = reference.to(torch.float32)
    diff = cand - ref
    k = min(int(topk), int(ref.shape[-1]))

    cand_idx = torch.topk(cand, k=k, dim=-1).indices
    ref_idx = torch.topk(ref, k=k, dim=-1).indices

    selected_mask = torch.zeros_like(ref, dtype=torch.bool).scatter_(-1, cand_idx, True)
    topk_overlap = torch.gather(selected_mask, -1, ref_idx).float().mean()

    top1_agreement = (
        torch.argmax(cand, dim=-1) == torch.argmax(ref, dim=-1)
    ).float().mean()

    ref_softmax = torch.softmax(ref, dim=-1)
    ref_mass_on_candidate_topk = torch.gather(
        ref_softmax,
        -1,
        cand_idx,
    ).sum(dim=-1).mean()

    return {
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        f"top{k}_overlap_vs_reference": float(topk_overlap.item()),
        "top1_agreement_vs_reference": float(top1_agreement.item()),
        f"reference_softmax_mass_on_candidate_top{k}": float(
            ref_mass_on_candidate_topk.item()
        ),
    }


@torch.no_grad()
def candidate_recall_metrics(
    *,
    full_logits: torch.Tensor,
    selected_indices: torch.Tensor,
    topk_quality: int,
) -> dict[str, float]:
    k = min(int(topk_quality), int(full_logits.shape[-1]))
    ref_topk_idx = torch.topk(full_logits, k=k, dim=-1).indices

    selected_mask = torch.zeros_like(full_logits, dtype=torch.bool).scatter_(
        -1,
        selected_indices,
        True,
    )
    candidate_recall = torch.gather(
        selected_mask,
        -1,
        ref_topk_idx,
    ).float().mean()

    full_softmax = torch.softmax(full_logits.to(torch.float32), dim=-1)
    selected_mass = torch.gather(
        full_softmax,
        -1,
        selected_indices,
    ).sum(dim=-1).mean()

    return {
        f"full_top{k}_candidate_recall": float(candidate_recall.item()),
        "full_softmax_mass_inside_selected_candidates": float(selected_mass.item()),
    }


@torch.no_grad()
def build_codebooks_and_sketch(
    *,
    device: torch.device,
    d: int,
    m: int,
    n_calib: int,
    seed: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x = torch.randn(n_calib, d, device=device, dtype=torch.float32)
    enc = recursive_polar_encode(x, num_levels=4)
    codebooks = fit_polar_angle_codebooks_from_encodings(
        [enc],
        bits_by_level=DEFAULT_POLAR_BITS_BY_LEVEL,
        max_iters=30,
        max_samples_per_level=200_000,
        seed=seed,
    )
    sketch = make_gaussian_sketch(
        d=d,
        m=m,
        device=device,
        dtype=torch.float32,
        seed=seed + 123,
    )
    return codebooks, sketch


@torch.no_grad()
def run_one(
    *,
    seq_len: int,
    args: argparse.Namespace,
    device: torch.device,
    codebooks,
    sketch: torch.Tensor,
) -> dict[str, Any]:
    B, H, Q, D, M = 1, int(args.num_heads), 1, 128, 128

    torch.manual_seed(args.seed + seq_len)
    torch.cuda.manual_seed_all(args.seed + seq_len)

    q = torch.randn(B, H, Q, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float32)

    encoding = turboquant_polar_prod_quantize(
        x=k,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=4,
    )

    packed = encoding.polar.packed_angles
    packed_qjl_signs = encoding.qjl_residual.packed_sign_bits.reshape(
        B,
        H,
        seq_len,
        M // 8,
    ).contiguous()
    qjl_norms = encoding.qjl_residual.norms.reshape(B, H, seq_len).contiguous()

    packed_meta = build_turboquant_packed_meta_blob(
        packed_l1=packed.level1_4bit,
        packed_l2=packed.level2_2bit,
        packed_l3=packed.level3_2bit,
        packed_l4=packed.level4_2bit,
        packed_qjl_signs=packed_qjl_signs,
    )

    q_projected = torch.matmul(
        q,
        sketch.T.to(torch.float32),
    ).contiguous()

    l2_factor_lut_fp16 = build_tree_l2_factor_lut(
        q=q,
        centroids_l1=codebooks.centroids[0],
        centroids_l2=codebooks.centroids[1],
    ).to(torch.float16).contiguous()

    lane_nibble_qjl_signs = build_qjl_lane_nibble_signs(
        packed_qjl_signs
    )

    common = dict(
        q_projected=q_projected,
        packed_meta=packed_meta,
        radii=encoding.polar.radii,
        l2_factor_lut_fp16=l2_factor_lut_fp16,
        centroids_l3=codebooks.centroids[2],
        centroids_l4=codebooks.centroids[3],
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        qjl_norms=qjl_norms,
    )

    def full_reference():
        return turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda(
            **common
        )

    def sparse_pipeline():
        r = selective_qjl_sparse_topk_m128_cuda(
            **common,
            topk=int(args.topk),
        )
        return r.polar_logits, r.selected_indices, r.selected_refined_logits

    def dense_pipeline():
        return selective_qjl_dense_logits_topk_m128_cuda(
            **common,
            topk=int(args.topk),
        )

    full_ms, full_logits = bench_cuda_ms(
        full_reference,
        warmup=args.warmup,
        iters=args.iters,
    )
    sparse_ms, sparse_out = bench_cuda_ms(
        sparse_pipeline,
        warmup=args.warmup,
        iters=args.iters,
    )
    dense_ms, dense_logits = bench_cuda_ms(
        dense_pipeline,
        warmup=args.warmup,
        iters=args.iters,
    )

    polar_logits, selected_indices, selected_refined_logits = sparse_out

    dense_for_quality = polar_logits.clone()
    dense_for_quality.scatter_(
        dim=-1,
        index=selected_indices,
        src=selected_refined_logits,
    )

    quality = score_metrics(
        dense_for_quality,
        full_logits,
        topk=int(args.quality_topk),
    )
    quality.update(
        candidate_recall_metrics(
            full_logits=full_logits,
            selected_indices=selected_indices,
            topk_quality=int(args.quality_topk),
        )
    )

    return {
        "seq_len": int(seq_len),
        "topk": int(min(int(args.topk), seq_len)),
        "timing_ms": {
            "full_fused_reference_ms": float(full_ms),
            "selective_sparse_pipeline_ms": float(sparse_ms),
            "selective_dense_materialized_pipeline_ms": float(dense_ms),
        },
        "speedup_vs_full_fused": {
            "sparse_pipeline_over_full_fused": float(full_ms / sparse_ms),
            "dense_materialized_pipeline_over_full_fused": float(full_ms / dense_ms),
        },
        "quality_vs_full_fused": quality,
        "representation_note": {
            "sparse_pipeline": (
                "Polar logits are dense; refined logits are sparse over selected top-K candidates."
            ),
            "dense_materialized_pipeline": (
                "For compatibility only: scatter refined top-K logits back into a dense Polar logit tensor."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Mainline benchmark for selective-QJL retrieval/refinement: "
            "Polar-only retrieval -> top-K candidate selection -> selected QJL refinement."
        )
    )
    p.add_argument("--seq_lens", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    p.add_argument("--topk", type=int, default=128)
    p.add_argument("--quality_topk", type=int, default=32)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--n_calib", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = ensure_cuda(args.device)

    print("========== Selective-QJL mainline benchmark ==========")
    print(args)
    print(
        "[Mainline] Polar-only retrieval -> top-K candidate selection -> selected QJL refinement."
    )
    print(
        "[Default] K=128, chosen from prior algorithmic CUDA experiments for the best "
        "speed/quality tradeoff on long contexts."
    )

    codebooks, sketch = build_codebooks_and_sketch(
        device=device,
        d=128,
        m=128,
        n_calib=int(args.n_calib),
        seed=int(args.seed),
    )

    results = []
    for seq_len in args.seq_lens:
        print("=" * 78)
        print(f"[Benchmark] T={int(seq_len)}")
        print("=" * 78)
        result = run_one(
            seq_len=int(seq_len),
            args=args,
            device=device,
            codebooks=codebooks,
            sketch=sketch,
        )
        print(json.dumps(result, indent=2))
        results.append(result)

    payload = {
        "benchmark": "selective_qjl_mainline",
        "method": (
            "polar_only_retrieval_then_torch_topk_candidate_selection_then_"
            "selected_qjl_refinement_cuda"
        ),
        "topk_selection_note": (
            "Top-K candidate selection currently uses torch.topk CUDA. "
            "This is the next bottleneck targeted for replacement by a custom selector."
        ),
        "config": vars(args),
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
