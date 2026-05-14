from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from turboquant.polarquant import PolarEncoding
from turboquant.polar_packing import (
    PackedPolarAngles,
    pack_polar_angle_codes_l4_d128,
    unpack_polar_angle_codes_l4_d128,
)

from turboquant.nvtx_utils import nvtx_range

from turboquant.polar_packing import (
    PackedPolarAngles,
    pack_polar_angle_codes_l4_d128,
    unpack_polar_angle_codes_l4_d128,
)

DEFAULT_POLAR_BITS_BY_LEVEL = (
    4,
    2,
    2,
    2,
)


@dataclass
class PolarAngleCodebooks:
    """
    Per-level angle codebooks.

    centroids[level_idx]:
        1D sorted centroid tensor for that polar level.

    bits_by_level[level_idx]:
        Bit-width used by that level.
    """
    centroids: list[torch.Tensor]
    bits_by_level: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.centroids) != len(self.bits_by_level):
            raise ValueError(
                "centroids and bits_by_level must have the same length."
            )

        for level_idx, (c, bits) in enumerate(
            zip(self.centroids, self.bits_by_level)
        ):
            expected = 1 << int(bits)

            if c.ndim != 1:
                raise ValueError(
                    f"centroids[{level_idx}] must be 1D."
                )

            if int(c.numel()) != expected:
                raise ValueError(
                    f"centroids[{level_idx}] has {c.numel()} entries, "
                    f"expected {expected} for {bits} bits."
                )


@dataclass
class QuantizedPolarEncoding:
    """
    Quantized Polar Stage-1 representation.

    Current paper-aligned main path:
        bits_by_level = (4,2,3,2)

    packed_angles:
        PackedPolarAnglesStage1ThreeBit

    radii:
        Remaining radii after L-level recursive polar transform.
        Stored as fp16.
    """
    packed_angles: PackedPolarAngles
    radii: torch.Tensor
    original_dim: int
    num_levels: int
    bits_by_level: tuple[int, ...]

def _flatten_float(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1).to(torch.float32)


@torch.no_grad()
def _fit_1d_kmeans_codebook(
    values: torch.Tensor,
    *,
    num_centroids: int,
    max_iters: int = 30,
    max_samples: int = 200_000,
    seed: int = 0,
) -> torch.Tensor:
    """
    Fit a 1D Lloyd k-means codebook.

    This is a reference implementation for PolarQuant Stage-1.
    It is intentionally simple and deterministic enough for testing.

    Args:
        values:
            Arbitrary-shape tensor of angle samples.
        num_centroids:
            Number of centroids, e.g. 16 for 4 bits, 4 for 2 bits.
        max_iters:
            Lloyd iterations.
        max_samples:
            Optional random subsampling cap to avoid large fitting cost.
        seed:
            Sampling seed.

    Returns:
        Sorted centroid tensor on the same device as values,
        dtype=torch.float32.
    """
    x = _flatten_float(values)

    if x.numel() == 0:
        raise ValueError("Cannot fit codebook on an empty tensor.")

    if int(x.numel()) > max_samples:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)

        perm = torch.randperm(
            int(x.numel()),
            generator=gen,
            device=x.device,
        )[:max_samples]

        x = x[perm]

    # Robust deterministic initialization using quantiles.
    q = torch.linspace(
        0.0,
        1.0,
        steps=num_centroids + 2,
        device=x.device,
        dtype=torch.float32,
    )[1:-1]

    centroids = torch.quantile(x, q).to(torch.float32)

    for _ in range(max_iters):
        distances = torch.abs(
            x.unsqueeze(-1) - centroids.unsqueeze(0)
        )

        assign = torch.argmin(
            distances,
            dim=-1,
        )

        new_centroids = centroids.clone()

        for k in range(num_centroids):
            mask = assign == k

            if bool(mask.any()):
                new_centroids[k] = x[mask].mean()

        new_centroids, _ = torch.sort(new_centroids)

        if torch.allclose(
            new_centroids,
            centroids,
            rtol=0.0,
            atol=1e-7,
        ):
            centroids = new_centroids
            break

        centroids = new_centroids

    return centroids


@torch.no_grad()
def fit_polar_angle_codebooks_from_encodings(
    encodings: Iterable[PolarEncoding],
    *,
    bits_by_level: Sequence[int] = DEFAULT_POLAR_BITS_BY_LEVEL,
    max_iters: int = 30,
    max_samples_per_level: int = 200_000,
    seed: int = 0,
) -> PolarAngleCodebooks:
    """
    Fit per-level angle codebooks from one or more PolarEncoding objects.

    This is the reference/calibration-backed codebook builder for
    TurboQuant Stage 1.

    In the eventual SVD-LLaMA pipeline, these samples should come from
    calibration key tensors, not from random synthetic tensors.
    """
    encodings = list(encodings)

    if len(encodings) == 0:
        raise ValueError("At least one PolarEncoding is required.")

    bits_by_level = tuple(int(b) for b in bits_by_level)
    num_levels = len(bits_by_level)

    for enc in encodings:
        if int(enc.num_levels) != num_levels:
            raise ValueError(
                f"Expected encoding.num_levels={num_levels}, "
                f"got {enc.num_levels}."
            )

    centroids: list[torch.Tensor] = []

    for level_idx, bits in enumerate(bits_by_level):
        angle_samples = torch.cat(
            [
                enc.angles[level_idx].reshape(-1).to(torch.float32)
                for enc in encodings
            ],
            dim=0,
        )

        level_centroids = _fit_1d_kmeans_codebook(
            angle_samples,
            num_centroids=1 << bits,
            max_iters=max_iters,
            max_samples=max_samples_per_level,
            seed=seed + level_idx,
        )

        centroids.append(level_centroids)

    return PolarAngleCodebooks(
        centroids=centroids,
        bits_by_level=bits_by_level,
    )


@torch.no_grad()
def quantize_angle_tensor(
    *,
    angle: torch.Tensor,
    centroids: torch.Tensor,
    max_chunk_elements: int = 4_000_000,
) -> torch.Tensor:
    """
    Quantize an angle tensor to nearest centroid.

    This chunked implementation avoids materializing a huge
    [..., num_centroids] distance tensor for long-context K caches.

    Args:
        angle:
            arbitrary shape angle tensor

        centroids:
            [C]

        max_chunk_elements:
            maximum number of flattened angle values processed
            in one chunk.

    Returns:
        codes:
            same shape as angle, dtype=torch.long
    """
    if centroids.ndim != 1:
        raise ValueError(
            f"centroids must be [C], got shape={tuple(centroids.shape)}"
        )

    if max_chunk_elements <= 0:
        raise ValueError(
            f"max_chunk_elements must be positive, got {max_chunk_elements}"
        )

    original_shape = angle.shape

    angle_flat = angle.reshape(-1).to(torch.float32)
    centroids_f = centroids.to(
        device=angle.device,
        dtype=torch.float32,
    )

    numel = int(angle_flat.numel())
    codes_flat = torch.empty(
        numel,
        device=angle.device,
        dtype=torch.long,
    )

    for start in range(0, numel, int(max_chunk_elements)):
        end = min(
            start + int(max_chunk_elements),
            numel,
        )

        chunk = angle_flat[start:end]

        # [chunk, C]
        distances = torch.abs(
            chunk.unsqueeze(-1) -
            centroids_f.unsqueeze(0)
        )

        codes_flat[start:end] = torch.argmin(
            distances,
            dim=-1,
        )

    return codes_flat.reshape(
        original_shape
    )

@torch.no_grad()
def dequantize_angle_codes(
    codes: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Restore angle tensor from integer centroid codes.
    """
    if codes.dtype not in (torch.int64, torch.long):
        codes = codes.to(torch.long)

    return centroids[codes]


@torch.no_grad()
def quantize_polar_encoding(
    encoding: PolarEncoding,
    codebooks: PolarAngleCodebooks,
) -> QuantizedPolarEncoding:
    """
    Quantize all angle levels, then pack them into byte storage.

    Current supported packed format:
      D = 128
      L = 4
      bits = (4,2,2,2)
    """
    if int(encoding.num_levels) != len(codebooks.centroids):
        raise ValueError(
            f"encoding.num_levels={encoding.num_levels} "
            f"does not match codebook levels={len(codebooks.centroids)}."
        )

    if int(encoding.original_dim) != 128:
        raise ValueError(
            "Packed Polar angle path currently requires original_dim=128."
        )

    if int(encoding.num_levels) != 4:
        raise ValueError(
            "Packed Polar angle path currently requires num_levels=4."
        )

    if tuple(codebooks.bits_by_level) != (4, 2, 2, 2):
        raise ValueError(
            "Packed Polar angle path currently requires "
            "bits_by_level=(4,2,2,2), "
            f"got {tuple(codebooks.bits_by_level)}."
        )

    angle_codes: list[torch.Tensor] = []

    for angle, centroids in zip(
        encoding.angles,
        codebooks.centroids,
    ):
        angle_codes.append(
            quantize_angle_tensor(
                angle=angle,
                centroids=centroids,
            )
        )

    packed_angles = pack_polar_angle_codes_l4_d128(
        angle_codes
    )

    return QuantizedPolarEncoding(
        packed_angles=packed_angles,
        radii=encoding.radii.to(torch.float16).contiguous(),
        original_dim=int(encoding.original_dim),
        num_levels=int(encoding.num_levels),
        bits_by_level=codebooks.bits_by_level,
    )

@torch.no_grad()
def dequantize_polar_encoding(
    qencoding: QuantizedPolarEncoding,
    codebooks: PolarAngleCodebooks,
) -> PolarEncoding:
    """
    Unpack angle codes, then restore dequantized angle tensors.
    """
    if int(qencoding.num_levels) != len(codebooks.centroids):
        raise ValueError(
            f"qencoding.num_levels={qencoding.num_levels} "
            f"does not match codebook levels={len(codebooks.centroids)}."
        )

    with nvtx_range("tq_polar_unpack_angles"):
        unpacked_angle_codes = unpack_polar_angle_codes_l4_d128(
            qencoding.packed_angles,
        )

    angles: list[torch.Tensor] = []

    with nvtx_range("tq_polar_dequantize_angles"):
        for codes, centroids in zip(
            unpacked_angle_codes,
            codebooks.centroids,
        ):
            angles.append(
                dequantize_angle_codes(
                    codes=codes,
                    centroids=centroids,
                )
            )

    return PolarEncoding(
        angles=angles,
        radii=qencoding.radii,
        original_dim=int(qencoding.original_dim),
        num_levels=int(qencoding.num_levels),
    )

def estimate_stage1_bits_per_coordinate(
    *,
    original_dim: int,
    num_levels: int,
    bits_by_level: Sequence[int],
    radius_bits: int = 16,
) -> float:
    """
    Estimate Stage-1 storage cost in bits/channel.

    For d=128, L=4, bits=[4,2,2,2], radius_bits=16:
      - angle bits:
          64*4 + 32*2 + 16*2 + 8*2 = 368 bits
      - remaining radii:
          8*16 = 128 bits
      - total:
          496 bits / 128 = 3.875 bits/channel
    """
    if original_dim <= 0:
        raise ValueError("original_dim must be positive.")

    if num_levels != len(bits_by_level):
        raise ValueError(
            "num_levels must match len(bits_by_level)."
        )

    total_angle_bits = 0

    for level_idx, bits in enumerate(bits_by_level, start=1):
        num_angles = original_dim // (2 ** level_idx)
        total_angle_bits += num_angles * int(bits)

    num_remaining_radii = original_dim // (2 ** num_levels)
    total_radius_bits = num_remaining_radii * int(radius_bits)

    total_bits = total_angle_bits + total_radius_bits

    return float(total_bits) / float(original_dim)
