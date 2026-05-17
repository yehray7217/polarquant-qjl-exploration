from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    root = Path(__file__).resolve().parent
    src = root / "csrc" / "turboquant_qjl_sign_layout_cuda.cu"

    _EXT = load(
        name="turboquant_qjl_sign_layout_cuda_ext",
        sources=[str(src)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
        ],
        extra_cflags=[
            "-O3",
        ],
        verbose=False,
    )
    return _EXT


def _validate_common(
    *,
    q_projected: torch.Tensor,
    qjl_sign_layout: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not q_projected.is_cuda:
        raise ValueError("q_projected must be CUDA.")
    if not qjl_sign_layout.is_cuda:
        raise ValueError("qjl_sign_layout must be CUDA.")
    if not qjl_norms.is_cuda:
        raise ValueError("qjl_norms must be CUDA.")

    return (
        q_projected.contiguous().to(torch.float32),
        qjl_sign_layout.contiguous(),
        qjl_norms.contiguous().to(torch.float16),
    )


@torch.no_grad()
def turboquant_qjl_only_split_sign_words_cuda(
    *,
    q_projected: torch.Tensor,
    packed_qjl_signs: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> torch.Tensor:
    """
    QJL-only ablation with signs split out of meta64.

    Input signs stay in the original sketch-major [B,H,T,16] byte layout.
    This isolates the effect of not reading the full 64-byte mixed metadata blob.
    """
    ext = _load_ext()
    qproj, signs, norms = _validate_common(
        q_projected=q_projected,
        qjl_sign_layout=packed_qjl_signs,
        qjl_norms=qjl_norms,
    )
    return ext.turboquant_qjl_only_split_sign_words_cuda(
        qproj,
        signs,
        norms,
    )


@torch.no_grad()
def turboquant_qjl_only_lane_nibble_signs_cuda(
    *,
    q_projected: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> torch.Tensor:
    """
    QJL-only ablation with lane-major nibble signs.

    Each lane reads one nibble containing its four QJL sign bits for
    sketch indices lane + 32*k, k=0..3.
    """
    ext = _load_ext()
    qproj, lane_signs, norms = _validate_common(
        q_projected=q_projected,
        qjl_sign_layout=lane_nibble_qjl_signs,
        qjl_norms=qjl_norms,
    )
    return ext.turboquant_qjl_only_lane_nibble_signs_cuda(
        qproj,
        lane_signs,
        norms,
    )
