from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
_EXT = None

def _load_ext():
    global _EXT
    if _EXT is None:
        src = Path(__file__).resolve().parent / 'csrc' / 'turboquant_algorithmic_reduced_qjl_cuda.cu'
        _EXT = load(name='turboquant_algorithmic_reduced_qjl_cuda_ext', sources=[str(src)], extra_cuda_cflags=['-O3','--use_fast_math'], extra_cflags=['-O3'], verbose=False)
    return _EXT

def _trig(centroids: torch.Tensor):
    c = centroids.contiguous().to(torch.float32)
    return torch.cos(c).contiguous(), torch.sin(c).contiguous()

@torch.no_grad()
def turboquant_fused_logits_compact_m64_cuda(*, q_projected, packed_meta, radii, l2_factor_lut_fp16, centroids_l3, centroids_l4, compact_qjl_signs, qjl_norms):
    ext = _load_ext(); cos_l3, sin_l3 = _trig(centroids_l3); cos_l4, sin_l4 = _trig(centroids_l4)
    return ext.turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_compact_m64_fused_logits_cuda(q_projected.contiguous().to(torch.float32), packed_meta.contiguous(), radii.contiguous().to(torch.float16), l2_factor_lut_fp16.contiguous().to(torch.float16), cos_l3, sin_l3, cos_l4, sin_l4, compact_qjl_signs.contiguous().to(torch.uint8), qjl_norms.contiguous().to(torch.float16))

@torch.no_grad()
def turboquant_fused_logits_compact_m32_cuda(*, q_projected, packed_meta, radii, l2_factor_lut_fp16, centroids_l3, centroids_l4, compact_qjl_signs, qjl_norms):
    ext = _load_ext(); cos_l3, sin_l3 = _trig(centroids_l3); cos_l4, sin_l4 = _trig(centroids_l4)
    return ext.turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_compact_m32_fused_logits_cuda(q_projected.contiguous().to(torch.float32), packed_meta.contiguous(), radii.contiguous().to(torch.float16), l2_factor_lut_fp16.contiguous().to(torch.float16), cos_l3, sin_l3, cos_l4, sin_l4, compact_qjl_signs.contiguous().to(torch.uint8), qjl_norms.contiguous().to(torch.float16))
