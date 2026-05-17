#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch
from turboquant.polarquant import recursive_polar_encode
from turboquant.polarquant_quant import DEFAULT_POLAR_BITS_BY_LEVEL, fit_polar_angle_codebooks_from_encodings
from turboquant.polar_prod import turboquant_polar_prod_quantize
from turboquant.qjl import make_gaussian_sketch
from turboquant.packed_meta import build_turboquant_packed_meta_blob
from turboquant.polar_tree_lut import build_tree_l2_factor_lut
from turboquant.qjl_sign_layout import build_qjl_lane_nibble_signs
from turboquant.qjl_compact_reduced_layout import build_qjl_compact_signs_m64, build_qjl_compact_signs_m32, pad_qjl_signs_to_meta64_16bytes
from turboquant.turboquant_qjl_norm_early_load_cuda import turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda
from turboquant.turboquant_algorithmic_reduced_qjl_cuda import turboquant_fused_logits_compact_m64_cuda, turboquant_fused_logits_compact_m32_cuda
from algo_bench_common import bench_cuda_ms, score_metrics, bytes_of

def ensure_cuda(s):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    return torch.device(s)

@torch.no_grad()
def build_codebooks(device, d, num_levels, n_calib, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    x = torch.randn(n_calib, d, device=device, dtype=torch.float32)
    enc = recursive_polar_encode(x, num_levels=num_levels)
    cb = fit_polar_angle_codebooks_from_encodings([enc], bits_by_level=DEFAULT_POLAR_BITS_BY_LEVEL, max_iters=30, max_samples_per_level=200_000, seed=seed)
    return cb

@torch.no_grad()
def run_one(seq_len, args, device, codebooks):
    B,H,Q,D = 1, args.num_heads, 1, 128
    torch.manual_seed(args.seed+seq_len); torch.cuda.manual_seed_all(args.seed+seq_len)
    q = torch.randn(B,H,Q,D, device=device, dtype=torch.float32)
    k = torch.randn(B,H,seq_len,D, device=device, dtype=torch.float32)
    sketches = {m: make_gaussian_sketch(d=D, m=m, device=device, dtype=torch.float32, seed=args.seed+123+m) for m in (128,64,32)}
    encs = {m: turboquant_polar_prod_quantize(x=k, codebooks=codebooks, sketch=sketches[m], num_levels=4) for m in (128,64,32)}
    l2_lut_fp16 = build_tree_l2_factor_lut(q=q, centroids_l1=codebooks.centroids[0], centroids_l2=codebooks.centroids[1]).to(torch.float16).contiguous()
    artifacts = {}
    for m in (128,64,32):
        enc = encs[m]; packed = enc.polar.packed_angles
        signs = enc.qjl_residual.packed_sign_bits.reshape(B,H,seq_len,m//8).contiguous()
        signs_meta = pad_qjl_signs_to_meta64_16bytes(signs)
        packed_meta = build_turboquant_packed_meta_blob(packed_l1=packed.level1_4bit, packed_l2=packed.level2_2bit, packed_l3=packed.level3_2bit, packed_l4=packed.level4_2bit, packed_qjl_signs=signs_meta)
        qproj = torch.matmul(q, sketches[m].T.to(torch.float32)).contiguous()
        norms = enc.qjl_residual.norms.reshape(B,H,seq_len).contiguous()
        artifacts[m] = dict(enc=enc, packed_meta=packed_meta, signs=signs, qproj=qproj, norms=norms)
    lane128 = build_qjl_lane_nibble_signs(artifacts[128]['signs'])
    compact64 = build_qjl_compact_signs_m64(artifacts[64]['signs'])
    compact32 = build_qjl_compact_signs_m32(artifacts[32]['signs'])
    def f128():
        return turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda(q_projected=artifacts[128]['qproj'], packed_meta=artifacts[128]['packed_meta'], radii=artifacts[128]['enc'].polar.radii, l2_factor_lut_fp16=l2_lut_fp16, centroids_l3=codebooks.centroids[2], centroids_l4=codebooks.centroids[3], lane_nibble_qjl_signs=lane128, qjl_norms=artifacts[128]['norms'])
    def f64():
        return turboquant_fused_logits_compact_m64_cuda(q_projected=artifacts[64]['qproj'], packed_meta=artifacts[64]['packed_meta'], radii=artifacts[64]['enc'].polar.radii, l2_factor_lut_fp16=l2_lut_fp16, centroids_l3=codebooks.centroids[2], centroids_l4=codebooks.centroids[3], compact_qjl_signs=compact64, qjl_norms=artifacts[64]['norms'])
    def f32():
        return turboquant_fused_logits_compact_m32_cuda(q_projected=artifacts[32]['qproj'], packed_meta=artifacts[32]['packed_meta'], radii=artifacts[32]['enc'].polar.radii, l2_factor_lut_fp16=l2_lut_fp16, centroids_l3=codebooks.centroids[2], centroids_l4=codebooks.centroids[3], compact_qjl_signs=compact32, qjl_norms=artifacts[32]['norms'])
    t128, out128 = bench_cuda_ms(f128, warmup=args.warmup, iters=args.iters)
    t64, out64 = bench_cuda_ms(f64, warmup=args.warmup, iters=args.iters)
    t32, out32 = bench_cuda_ms(f32, warmup=args.warmup, iters=args.iters)
    return {
        'seq_len': int(seq_len),
        'timing_ms': {'m128_reference_ms': t128, 'm64_compact_ms': t64, 'm32_compact_ms': t32},
        'speedup_vs_m128': {'m64_over_m128': t128/t64, 'm32_over_m128': t128/t32},
        'quality_vs_m128_score': {'m64': score_metrics(out64, out128, topk=args.quality_topk), 'm32': score_metrics(out32, out128, topk=args.quality_topk)},
        'qjl_sign_storage_bytes': {'m128_lane_nibble_bytes': bytes_of(lane128), 'm64_compact_bytes': bytes_of(compact64), 'm32_compact_bytes': bytes_of(compact32)},
    }

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seq_lens', type=int, nargs='+', default=[16384,32768,65536,131072]); p.add_argument('--warmup', type=int, default=20); p.add_argument('--iters', type=int, default=100); p.add_argument('--out', required=True); p.add_argument('--device', default='cuda:0'); p.add_argument('--seed', type=int, default=0); p.add_argument('--num_heads', type=int, default=32); p.add_argument('--n_calib', type=int, default=4096); p.add_argument('--quality_topk', type=int, default=32); args=p.parse_args()
    dev=ensure_cuda(args.device); print('========== Algorithmic CUDA: reduced QJL dimension M sweep ==========' ); print(args)
    cb=build_codebooks(dev,128,4,args.n_calib,args.seed)
    results=[]
    for s in args.seq_lens:
        print('='*78); print(f'[Benchmark] T={s}'); print('='*78); r=run_one(int(s),args,dev,cb); print(json.dumps(r,indent=2)); results.append(r)
    payload={'benchmark':'algorithmic_cuda_reduced_qjl_dimension', 'method':'m128_reference_vs_compact_m64_vs_compact_m32', 'config':vars(args), 'results':results}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)); print(f'[Save] {out}')
if __name__=='__main__': main()
