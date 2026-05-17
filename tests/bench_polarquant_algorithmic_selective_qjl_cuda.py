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
from turboquant.turboquant_radii_access_optimization_cuda import turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda
from turboquant.turboquant_qjl_norm_early_load_cuda import turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda
from turboquant.turboquant_selected_qjl_refine_cuda import turboquant_selected_qjl_refine_topk_m128_cuda
from algo_bench_common import bench_cuda_ms, score_metrics

def ensure_cuda(s):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    return torch.device(s)

@torch.no_grad()
def build_codebooks_sketch(device,d,m,n_calib,seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    x=torch.randn(n_calib,d,device=device,dtype=torch.float32); enc=recursive_polar_encode(x,num_levels=4)
    cb=fit_polar_angle_codebooks_from_encodings([enc], bits_by_level=DEFAULT_POLAR_BITS_BY_LEVEL, max_iters=30, max_samples_per_level=200_000, seed=seed)
    sketch=make_gaussian_sketch(d=d,m=m,device=device,dtype=torch.float32,seed=seed+123)
    return cb, sketch

@torch.no_grad()
def selective_quality(full_logits, polar_logits, selected_idx, selected_refined, topk_quality):
    approx = polar_logits.clone().scatter(-1, selected_idx, selected_refined)
    metric = score_metrics(approx, full_logits, topk=topk_quality)
    p = torch.softmax(full_logits.to(torch.float32), dim=-1)
    selected_mass = torch.gather(p, -1, selected_idx).sum(dim=-1).mean()
    full_topk_idx = torch.topk(full_logits, k=min(topk_quality, full_logits.shape[-1]), dim=-1).indices
    selected_mask = torch.zeros_like(full_logits, dtype=torch.bool).scatter(-1, selected_idx, True)
    full_topk_candidate_recall = torch.gather(selected_mask, -1, full_topk_idx).float().mean()
    metric.update({'full_softmax_mass_inside_selected_candidates': float(selected_mass.item()), f'full_top{min(topk_quality, full_logits.shape[-1])}_candidate_recall': float(full_topk_candidate_recall.item())})
    return metric

@torch.no_grad()
def run_one(seq_len,args,device,cb,sketch):
    B,H,Q,D,M=1,args.num_heads,1,128,128
    torch.manual_seed(args.seed+seq_len); torch.cuda.manual_seed_all(args.seed+seq_len)
    q=torch.randn(B,H,Q,D,device=device,dtype=torch.float32); k=torch.randn(B,H,seq_len,D,device=device,dtype=torch.float32)
    enc=turboquant_polar_prod_quantize(x=k, codebooks=cb, sketch=sketch, num_levels=4)
    packed=enc.polar.packed_angles; signs=enc.qjl_residual.packed_sign_bits.reshape(B,H,seq_len,16).contiguous(); norms=enc.qjl_residual.norms.reshape(B,H,seq_len).contiguous()
    packed_meta=build_turboquant_packed_meta_blob(packed_l1=packed.level1_4bit, packed_l2=packed.level2_2bit, packed_l3=packed.level3_2bit, packed_l4=packed.level4_2bit, packed_qjl_signs=signs)
    qproj=torch.matmul(q,sketch.T.to(torch.float32)).contiguous(); lut=build_tree_l2_factor_lut(q=q,centroids_l1=cb.centroids[0],centroids_l2=cb.centroids[1]).to(torch.float16).contiguous(); lane=build_qjl_lane_nibble_signs(signs)
    def full(): return turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda(q_projected=qproj, packed_meta=packed_meta, radii=enc.polar.radii, l2_factor_lut_fp16=lut, centroids_l3=cb.centroids[2], centroids_l4=cb.centroids[3], lane_nibble_qjl_signs=lane, qjl_norms=norms)
    def polar(): return turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda(packed_meta=packed_meta, radii=enc.polar.radii, l2_factor_lut_fp16=lut, centroids_l3=cb.centroids[2], centroids_l4=cb.centroids[3])
    tfull, full_out = bench_cuda_ms(full, warmup=args.warmup, iters=args.iters)
    tpolar, polar_out = bench_cuda_ms(polar, warmup=args.warmup, iters=args.iters)
    ks=[]
    for K in args.topk_candidates:
        K=min(int(K),seq_len)
        # Build once for quality.
        _, idx = torch.topk(polar_out, k=K, dim=-1)
        refined = turboquant_selected_qjl_refine_topk_m128_cuda(q_projected=qproj, lane_nibble_qjl_signs=lane, qjl_norms=norms, polar_logits=polar_out, selected_indices=idx)
        # Timed pipeline: Polar -> torch.topk CUDA -> custom selected QJL refine.
        def pipe():
            pscore = polar()
            _, sel = torch.topk(pscore, k=K, dim=-1)
            return turboquant_selected_qjl_refine_topk_m128_cuda(q_projected=qproj, lane_nibble_qjl_signs=lane, qjl_norms=norms, polar_logits=pscore, selected_indices=sel)
        tpipe, _ = bench_cuda_ms(pipe, warmup=args.warmup, iters=args.iters)
        # Refinement-only time with precomputed indices/polar.
        def refine_only(): return turboquant_selected_qjl_refine_topk_m128_cuda(q_projected=qproj, lane_nibble_qjl_signs=lane, qjl_norms=norms, polar_logits=polar_out, selected_indices=idx)
        tref, _ = bench_cuda_ms(refine_only, warmup=args.warmup, iters=args.iters)
        ks.append({'K':K, 'timing_ms': {'polar_only_full_scan_ms':tpolar, 'selected_qjl_refine_only_ms':tref, 'polar_topk_refine_pipeline_ms':tpipe, 'full_fused_reference_ms':tfull}, 'speedup_vs_full_fused': {'pipeline_over_full_fused':tfull/tpipe}, 'quality_vs_full_fused': selective_quality(full_out,polar_out,idx,refined,args.quality_topk)})
    return {'seq_len':seq_len, 'full_fused_reference_ms':tfull, 'polar_only_ms':tpolar, 'candidate_results':ks}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seq_lens',type=int,nargs='+',default=[16384,32768,65536,131072]); p.add_argument('--topk_candidates',type=int,nargs='+',default=[128,256,512,1024]); p.add_argument('--quality_topk',type=int,default=32); p.add_argument('--warmup',type=int,default=20); p.add_argument('--iters',type=int,default=100); p.add_argument('--out',required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=0); p.add_argument('--num_heads',type=int,default=32); p.add_argument('--n_calib',type=int,default=4096); args=p.parse_args()
    dev=ensure_cuda(args.device); print('========== Algorithmic CUDA: Polar top-K selective QJL refinement ==========' ); print(args)
    cb,sketch=build_codebooks_sketch(dev,128,128,args.n_calib,args.seed)
    results=[]
    for s in args.seq_lens:
        print('='*78); print(f'[Benchmark] T={s}'); print('='*78); r=run_one(int(s),args,dev,cb,sketch); print(json.dumps(r,indent=2)); results.append(r)
    payload={'benchmark':'algorithmic_cuda_selective_qjl_refinement', 'method':'polar_full_scan_torch_topk_cuda_custom_selected_qjl_refine', 'topk_note':'Top-K selection uses torch.topk CUDA; the selected QJL refinement stage is a custom CUDA kernel.', 'config':vars(args), 'results':results}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)); print(f'[Save] {out}')
if __name__=='__main__': main()
