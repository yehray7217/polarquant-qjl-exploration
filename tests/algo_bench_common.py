from __future__ import annotations
import math
from typing import Callable
import torch

def sync(): torch.cuda.synchronize()

@torch.no_grad()
def bench_cuda_ms(fn: Callable, *, warmup: int, iters: int):
    out = None
    for _ in range(warmup): out = fn()
    sync(); start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); start.record()
    for _ in range(iters): out = fn()
    end.record(); sync()
    return float(start.elapsed_time(end)/iters), out

@torch.no_grad()
def score_metrics(candidate: torch.Tensor, reference: torch.Tensor, topk: int = 32):
    c = candidate.to(torch.float32); r = reference.to(torch.float32)
    diff = (c-r).abs()
    rmse = torch.sqrt(torch.mean((c-r)**2))
    k = min(int(topk), int(r.shape[-1]))
    ref_idx = torch.topk(r, k=k, dim=-1).indices
    cand_idx = torch.topk(c, k=k, dim=-1).indices
    # top-k overlap, averaged over B,H,Q
    cand_mask = torch.zeros_like(r, dtype=torch.bool).scatter(-1, cand_idx, True)
    overlap = torch.gather(cand_mask, -1, ref_idx).float().mean()
    ref_top1 = torch.argmax(r, dim=-1)
    cand_top1 = torch.argmax(c, dim=-1)
    top1_agree = (ref_top1 == cand_top1).float().mean()
    p_ref = torch.softmax(r, dim=-1)
    mass_on_cand_topk = torch.gather(p_ref, -1, cand_idx).sum(dim=-1).mean()
    return {
        'max_abs_diff': float(diff.max().item()),
        'mean_abs_diff': float(diff.mean().item()),
        'rmse': float(rmse.item()),
        f'top{k}_overlap_vs_reference': float(overlap.item()),
        'top1_agreement_vs_reference': float(top1_agree.item()),
        f'reference_softmax_mass_on_candidate_top{k}': float(mass_on_cand_topk.item()),
    }

def bytes_of(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())
