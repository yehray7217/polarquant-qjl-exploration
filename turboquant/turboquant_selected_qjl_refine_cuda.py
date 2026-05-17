from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
_EXT = None

def _load_ext():
    global _EXT
    if _EXT is None:
        src = Path(__file__).resolve().parent / 'csrc' / 'turboquant_selected_qjl_refine_cuda.cu'
        _EXT = load(name='turboquant_selected_qjl_refine_cuda_ext', sources=[str(src)], extra_cuda_cflags=['-O3','--use_fast_math'], extra_cflags=['-O3'], verbose=False)
    return _EXT

@torch.no_grad()
def turboquant_selected_qjl_refine_topk_m128_cuda(*, q_projected, lane_nibble_qjl_signs, qjl_norms, polar_logits, selected_indices):
    ext = _load_ext()
    return ext.turboquant_selected_qjl_refine_topk_m128_cuda(q_projected.contiguous().to(torch.float32), lane_nibble_qjl_signs.contiguous().to(torch.uint8), qjl_norms.contiguous().to(torch.float16), polar_logits.contiguous().to(torch.float32), selected_indices.contiguous().to(torch.int64))
