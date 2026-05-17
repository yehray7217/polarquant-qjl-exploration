#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for p in (str(ROOT), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import bench_polarquant_selective_qjl_mainline as base
from turboquant.polar_prod import turboquant_polar_prod_quantize
from turboquant.packed_meta import build_turboquant_packed_meta_blob
from turboquant.polar_tree_lut import build_tree_l2_factor_lut
from turboquant.qjl_sign_layout import build_qjl_lane_nibble_signs
from turboquant.selective_qjl_custom_selector_pipeline import (
    selective_qjl_custom_selector_sparse_topk_m128_cuda,
    selective_qjl_custom_selector_dense_logits_topk_m128_cuda,
)
from turboquant.selective_qjl_pipeline import (
    selective_qjl_sparse_topk_m128_cuda,
)
from turboquant.turboquant_qjl_norm_early_load_cuda import (
    turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda,
)


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
        B, H, seq_len, M // 8,
    ).contiguous()
    qjl_norms = encoding.qjl_residual.norms.reshape(B, H, seq_len).contiguous()

    packed_meta = build_turboquant_packed_meta_blob(
        packed_l1=packed.level1_4bit,
        packed_l2=packed.level2_2bit,
        packed_l3=packed.level3_2bit,
        packed_l4=packed.level4_2bit,
        packed_qjl_signs=packed_qjl_signs,
    )
    q_projected = torch.matmul(q, sketch.T.to(torch.float32)).contiguous()
    l2_factor_lut_fp16 = build_tree_l2_factor_lut(
        q=q,
        centroids_l1=codebooks.centroids[0],
        centroids_l2=codebooks.centroids[1],
    ).to(torch.float16).contiguous()
    lane_nibble_qjl_signs = build_qjl_lane_nibble_signs(packed_qjl_signs)

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

    def torch_topk_sparse():
        r = selective_qjl_sparse_topk_m128_cuda(**common, topk=int(args.topk))
        return r.polar_logits, r.selected_indices, r.selected_refined_logits

    def custom_sparse():
        r = selective_qjl_custom_selector_sparse_topk_m128_cuda(
            **common,
            topk=int(args.topk),
        )
        return r.polar_logits, r.selected_indices, r.selected_refined_logits

    def custom_dense():
        return selective_qjl_custom_selector_dense_logits_topk_m128_cuda(
            **common,
            topk=int(args.topk),
        )

    full_ms, full_logits = base.bench_cuda_ms(
        full_reference, warmup=args.warmup, iters=args.iters
    )
    torch_sparse_ms, torch_sparse_out = base.bench_cuda_ms(
        torch_topk_sparse, warmup=args.warmup, iters=args.iters
    )
    custom_sparse_ms, custom_sparse_out = base.bench_cuda_ms(
        custom_sparse, warmup=args.warmup, iters=args.iters
    )
    custom_dense_ms, custom_dense_logits = base.bench_cuda_ms(
        custom_dense, warmup=args.warmup, iters=args.iters
    )

    polar_logits, custom_indices, custom_refined = custom_sparse_out
    custom_dense_for_quality = polar_logits.clone()
    custom_dense_for_quality.scatter_(
        dim=-1,
        index=custom_indices,
        src=custom_refined,
    )

    quality = base.score_metrics(
        custom_dense_for_quality,
        full_logits,
        topk=int(args.quality_topk),
    )
    quality.update(
        base.candidate_recall_metrics(
            full_logits=full_logits,
            selected_indices=custom_indices,
            topk_quality=int(args.quality_topk),
        )
    )

    return {
        "seq_len": int(seq_len),
        "topk": int(args.topk),
        "timing_ms": {
            "full_fused_reference_ms": float(full_ms),
            "torch_topk_sparse_pipeline_ms": float(torch_sparse_ms),
            "custom_selector_sparse_pipeline_ms": float(custom_sparse_ms),
            "custom_selector_dense_materialized_pipeline_ms": float(custom_dense_ms),
        },
        "speedup_vs_full_fused": {
            "torch_topk_sparse_over_full_fused": float(full_ms / torch_sparse_ms),
            "custom_selector_sparse_over_full_fused": float(full_ms / custom_sparse_ms),
            "custom_selector_dense_over_full_fused": float(full_ms / custom_dense_ms),
        },
        "speedup_vs_torch_topk_sparse": {
            "custom_selector_sparse_over_torch_topk_sparse": float(
                torch_sparse_ms / custom_sparse_ms
            ),
        },
        "quality_vs_full_fused": quality,
        "selector_note": (
            "Custom selector = local warp top-8 over each 128-token group, "
            "then exact per-head merge top-K over the pooled candidates."
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Custom CUDA top-K selector benchmark for selective-QJL."
    )
    p.add_argument("--seq_lens", type=int, nargs="+", default=[32768, 65536, 131072])
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


def main():
    args = parse_args()
    device = base.ensure_cuda(args.device)
    print("========== Selective-QJL custom CUDA candidate selector benchmark ==========")
    print(args)

    codebooks, sketch = base.build_codebooks_and_sketch(
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
        "benchmark": "selective_qjl_custom_cuda_candidate_selector",
        "method": (
            "polar_retrieval_then_custom_two_stage_candidate_selector_then_"
            "selected_qjl_refinement"
        ),
        "config": vars(args),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
