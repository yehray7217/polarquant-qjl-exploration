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
    src = root / "csrc" / "turboquant_polar_l1_lut_stall_probe_cuda.cu"

    _EXT = load(
        name="turboquant_polar_l1_lut_stall_probe_cuda_ext",
        sources=[str(src)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _EXT


def _trig(centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = centroids.contiguous().to(torch.float32)
    return torch.cos(c).contiguous(), torch.sin(c).contiguous()


def _common_args(
    *,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    l1_factor_lut: torch.Tensor,
    centroids_l2: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
):
    for name, tensor in {
        "packed_meta": packed_meta,
        "radii": radii,
        "l1_factor_lut": l1_factor_lut,
    }.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA.")

    cos_l2, sin_l2 = _trig(centroids_l2)
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return (
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        l1_factor_lut.contiguous().to(torch.float32),
        cos_l2,
        sin_l2,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
    )


@torch.no_grad()
def turboquant_polar_l1_lut_no_factor_global_load_cuda(
    *,
    packed_meta: torch.Tensor,
    radii: torch.Tensor,
    centroids_l2: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
) -> torch.Tensor:
    """
    Probe A: remove the L1 factor LUT global loads.

    The kernel keeps packed angle-code decode, downstream tree combines, and
    radii loads, but substitutes deterministic register-only synthetic s1
    factors derived from c1a/c1b. Output is NOT meant to match the baseline;
    only latency / NCU counters are meaningful.
    """
    ext = _load_ext()

    if not packed_meta.is_cuda or not radii.is_cuda:
        raise ValueError("packed_meta and radii must be CUDA.")

    cos_l2, sin_l2 = _trig(centroids_l2)
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_l1_lut_no_factor_global_load_cuda(
        packed_meta.contiguous(),
        radii.contiguous().to(torch.float16),
        cos_l2,
        sin_l2,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
    )


@torch.no_grad()
def turboquant_polar_l1_lut_no_radii_global_load_cuda(
    *,
    packed_meta: torch.Tensor,
    l1_factor_lut: torch.Tensor,
    centroids_l2: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
) -> torch.Tensor:
    """
    Probe B: remove radii global loads.

    The kernel keeps packed angle-code decode and factor LUT loads, but uses
    radius=1.0f in the final Polar block contribution. Output is NOT meant
    to match the baseline; only latency / NCU counters are meaningful.
    """
    ext = _load_ext()

    if not packed_meta.is_cuda or not l1_factor_lut.is_cuda:
        raise ValueError("packed_meta and l1_factor_lut must be CUDA.")

    cos_l2, sin_l2 = _trig(centroids_l2)
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_l1_lut_no_radii_global_load_cuda(
        packed_meta.contiguous(),
        l1_factor_lut.contiguous().to(torch.float32),
        cos_l2,
        sin_l2,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
    )


@torch.no_grad()
def turboquant_polar_l1_lut_direct_u8_codes_cuda(
    *,
    l1_codes: torch.Tensor,
    l2_codes: torch.Tensor,
    l3_codes: torch.Tensor,
    l4_codes: torch.Tensor,
    radii: torch.Tensor,
    l1_factor_lut: torch.Tensor,
    centroids_l2: torch.Tensor,
    centroids_l3: torch.Tensor,
    centroids_l4: torch.Tensor,
) -> torch.Tensor:
    """
    Probe C: replace packed meta angle-code unpacking with direct uint8 codes.

    This removes bit unpacking + meta-word shuffle decode, but intentionally
    introduces widened direct code tensors. Output SHOULD match the baseline
    Polar-only L1-LUT kernel; this probe is useful for deciding whether angle
    decode itself is worth optimizing.
    """
    ext = _load_ext()

    for name, tensor in {
        "l1_codes": l1_codes,
        "l2_codes": l2_codes,
        "l3_codes": l3_codes,
        "l4_codes": l4_codes,
        "radii": radii,
        "l1_factor_lut": l1_factor_lut,
    }.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA.")

    cos_l2, sin_l2 = _trig(centroids_l2)
    cos_l3, sin_l3 = _trig(centroids_l3)
    cos_l4, sin_l4 = _trig(centroids_l4)

    return ext.turboquant_polar_l1_lut_direct_u8_codes_cuda(
        l1_codes.contiguous(),
        l2_codes.contiguous(),
        l3_codes.contiguous(),
        l4_codes.contiguous(),
        radii.contiguous().to(torch.float16),
        l1_factor_lut.contiguous().to(torch.float32),
        cos_l2,
        sin_l2,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
    )
