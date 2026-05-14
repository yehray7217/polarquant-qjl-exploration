from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from turboquant.nvtx_utils import (
    nvtx_range,
)
from turboquant.polar_packing import (
    PackedPolarAngles,
    packed_polar_angle_storage_bytes,
)
from turboquant.polar_prod import (
    PolarProdEncoding,
    turboquant_polar_prod_quantize,
    turboquant_polar_prod_score_against_chunk,
)
from turboquant.polarquant_quant import (
    PolarAngleCodebooks,
    QuantizedPolarEncoding,
)
from turboquant.qjl import (
    QJLEncoding,
)
from turboquant.qjl_packing import (
    packed_qjl_sign_storage_bytes,
)


def _tensor_bytes(x: torch.Tensor) -> int:
    return int(
        x.numel() *
        x.element_size()
    )


@dataclass
class PolarProdLayerCache:
    """
    Chunked compressed K cache for one transformer layer.

    chunks:
        List of PolarProdEncoding chunks.

    chunk_seq_lens:
        Token count per chunk.

    seen_tokens:
        Total cached tokens for this layer.
    """
    chunks: list[PolarProdEncoding] = field(
        default_factory=list
    )
    chunk_seq_lens: list[int] = field(
        default_factory=list
    )
    seen_tokens: int = 0


class PolarProdKeyCache:
    """
    TurboQuant PolarProd compressed K-cache.

    Stored K representation:
      - Packed Polar Stage-1 angle codes
      - Polar residual radii
      - Packed 1-bit QJL residual signs
      - QJL residual norms

    Runtime scoring:
      - Fused Polar Stage-1 CUDA score
      - Fused QJL packed-sign CUDA score

    The cache is kept as token chunks. To avoid one chunk per decode token,
    small tail chunks are merged up to decode_tail_chunk_size.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        codebooks: PolarAngleCodebooks,
        sketch: torch.Tensor,
        num_levels: int = 4,
        decode_tail_chunk_size: int = 128,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if decode_tail_chunk_size <= 0:
            raise ValueError(
                "decode_tail_chunk_size must be positive."
            )

        self.num_layers = int(num_layers)
        self.codebooks = codebooks
        self.sketch = sketch
        self.num_levels = int(num_levels)
        self.decode_tail_chunk_size = int(
            decode_tail_chunk_size
        )

        self.layers: list[PolarProdLayerCache] = [
            PolarProdLayerCache()
            for _ in range(self.num_layers)
        ]

    def _check_layer_idx(
        self,
        layer_idx: int,
    ) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx out of range: {layer_idx}, "
                f"num_layers={self.num_layers}"
            )

    @torch.no_grad()
    def _merge_two_encodings_along_token_axis(
        self,
        left: PolarProdEncoding,
        right: PolarProdEncoding,
    ) -> PolarProdEncoding:
        """
        Merge two compressed K chunks along token axis.

        Expected original logical layouts:
          left:  [B,H,T_left,D]
          right: [B,H,T_right,D]
        """
        left_packed = left.polar.packed_angles
        right_packed = right.polar.packed_angles

        merged_packed_angles = PackedPolarAngles(
            level1_4bit=torch.cat(
                [
                    left_packed.level1_4bit,
                    right_packed.level1_4bit,
                ],
                dim=2,
            ).contiguous(),
            level2_2bit=torch.cat(
                [
                    left_packed.level2_2bit,
                    right_packed.level2_2bit,
                ],
                dim=2,
            ).contiguous(),
            level3_2bit=torch.cat(
                [
                    left_packed.level3_2bit,
                    right_packed.level3_2bit,
                ],
                dim=2,
            ).contiguous(),
            level4_2bit=torch.cat(
                [
                    left_packed.level4_2bit,
                    right_packed.level4_2bit,
                ],
                dim=2,
            ).contiguous(),
        )

        merged_radii = torch.cat(
            [
                left.polar.radii,
                right.polar.radii,
            ],
            dim=2,
        ).contiguous()

        merged_polar = QuantizedPolarEncoding(
            packed_angles=merged_packed_angles,
            radii=merged_radii,
            original_dim=int(
                left.polar.original_dim
            ),
            num_levels=int(
                left.polar.num_levels
            ),
            bits_by_level=left.polar.bits_by_level,
        )

        B = int(merged_radii.shape[0])
        H = int(merged_radii.shape[1])

        T_left = int(
            left.polar.radii.shape[2]
        )
        T_right = int(
            right.polar.radii.shape[2]
        )
        T_total = T_left + T_right

        # ------------------------------------------------------------
        # Merge QJL packed signs.
        #
        # Stored flattened layout:
        #   [B*H*T, packed_M]
        #
        # Reshape back to:
        #   [B,H,T,packed_M]
        # concatenate token dim,
        # flatten again.
        # ------------------------------------------------------------

        left_signs = (
            left.qjl_residual.packed_sign_bits.reshape(
                B,
                H,
                T_left,
                -1,
            )
        )

        right_signs = (
            right.qjl_residual.packed_sign_bits.reshape(
                B,
                H,
                T_right,
                -1,
            )
        )

        merged_signs = torch.cat(
            [
                left_signs,
                right_signs,
            ],
            dim=2,
        ).contiguous().reshape(
            B * H * T_total,
            -1,
        )

        # ------------------------------------------------------------
        # Merge QJL norms.
        #
        # Stored flattened layout:
        #   [B*H*T]
        # ------------------------------------------------------------

        left_norms = left.qjl_residual.norms.reshape(
            B,
            H,
            T_left,
        )

        right_norms = right.qjl_residual.norms.reshape(
            B,
            H,
            T_right,
        )

        merged_norms = torch.cat(
            [
                left_norms,
                right_norms,
            ],
            dim=2,
        ).contiguous().reshape(
            B * H * T_total,
        )

        merged_qjl = QJLEncoding(
            packed_sign_bits=merged_signs,
            norms=merged_norms,
        )

        return PolarProdEncoding(
            polar=merged_polar,
            qjl_residual=merged_qjl,
        )

    @torch.no_grad()
    def append(
        self,
        *,
        layer_idx: int,
        key_states: torch.Tensor,
    ) -> None:
        """
        Append one or more K tokens.

        Args:
            layer_idx:
                Transformer layer index.

            key_states:
                [B,H,T_new,D]
        """
        self._check_layer_idx(layer_idx)

        if key_states.ndim != 4:
            raise ValueError(
                "key_states must be [B,H,T,D], "
                f"got shape={tuple(key_states.shape)}"
            )

        T_new = int(
            key_states.shape[2]
        )

        if T_new <= 0:
            raise ValueError(
                "Cannot append an empty K chunk."
            )

        encoding = turboquant_polar_prod_quantize(
            x=key_states.to(torch.float32),
            codebooks=self.codebooks,
            sketch=self.sketch,
            num_levels=self.num_levels,
        )

        layer = self.layers[layer_idx]

        # ------------------------------------------------------------
        # Decode-tail chunk merging.
        #
        # If the newest chunk is small and the current tail chunk is
        # still below decode_tail_chunk_size, merge them.
        #
        # For long prefill prompts, prefill chunk is typically much
        # larger than decode_tail_chunk_size and therefore remains as-is.
        # ------------------------------------------------------------

        should_merge_into_tail = (
            len(layer.chunks) > 0
            and T_new <= self.decode_tail_chunk_size
            and layer.chunk_seq_lens[-1]
            < self.decode_tail_chunk_size
        )

        if should_merge_into_tail:
            old_tail_len = int(
                layer.chunk_seq_lens[-1]
            )

            if (
                old_tail_len + T_new
                <= self.decode_tail_chunk_size
            ):
                merged = (
                    self._merge_two_encodings_along_token_axis(
                        layer.chunks[-1],
                        encoding,
                    )
                )

                layer.chunks[-1] = merged
                layer.chunk_seq_lens[-1] = (
                    old_tail_len + T_new
                )
            else:
                layer.chunks.append(encoding)
                layer.chunk_seq_lens.append(T_new)
        else:
            layer.chunks.append(encoding)
            layer.chunk_seq_lens.append(T_new)

        layer.seen_tokens += T_new

    @torch.no_grad()
    def score(
        self,
        *,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute compressed-domain attention scores.

        Args:
            layer_idx:
                Transformer layer index.

            query_states:
                [B,H,Q,D]

        Returns:
            [B,H,Q,T_cache]
        """
        self._check_layer_idx(layer_idx)

        if query_states.ndim != 4:
            raise ValueError(
                "query_states must be [B,H,Q,D], "
                f"got shape={tuple(query_states.shape)}"
            )

        Q = int(
            query_states.shape[2]
        )

        if Q <= 0:
            raise ValueError(
                f"query_states must have Q >= 1, got Q={Q}"
            )

        layer = self.layers[layer_idx]

        if len(layer.chunks) == 0:
            raise RuntimeError(
                f"Layer {layer_idx} cache is empty."
            )

        with nvtx_range("tq_polar_score_total"):
            per_chunk_score_blocks: list[torch.Tensor] = []

            for encoding in layer.chunks:
                with nvtx_range("tq_polar_score_one_chunk"):
                    chunk_scores = (
                        turboquant_polar_prod_score_against_chunk(
                            q=query_states,
                            encoding=encoding,
                            codebooks=self.codebooks,
                            sketch=self.sketch,
                        )
                    )

                # [B,H,Q,T_chunk]
                per_chunk_score_blocks.append(
                    chunk_scores
                )

            return torch.cat(
                per_chunk_score_blocks,
                dim=-1,
            )

    def seq_len(
        self,
        *,
        layer_idx: int,
    ) -> int:
        self._check_layer_idx(layer_idx)
        return int(
            self.layers[layer_idx].seen_tokens
        )

    def storage_bytes(
        self,
    ) -> int:
        """
        Count actual stored compressed-K bytes.

        Includes:
          - packed Polar angle bytes
          - Polar radii
          - packed QJL sign bytes
          - QJL norms
        """
        total = 0

        for layer in self.layers:
            for encoding in layer.chunks:
                total += packed_polar_angle_storage_bytes(
                    encoding.polar.packed_angles
                )

                total += _tensor_bytes(
                    encoding.polar.radii
                )

                total += packed_qjl_sign_storage_bytes(
                    encoding.qjl_residual.packed_sign_bits
                )

                total += _tensor_bytes(
                    encoding.qjl_residual.norms
                )

        return int(total)

    def report(
        self,
    ) -> dict[str, Any]:
        layer_reports: list[dict[str, Any]] = []

        for layer_idx, layer in enumerate(self.layers):
            layer_reports.append(
                {
                    "layer_idx": int(layer_idx),
                    "seen_tokens": int(layer.seen_tokens),
                    "num_chunks": int(len(layer.chunks)),
                    "chunk_seq_lens": [
                        int(x)
                        for x in layer.chunk_seq_lens
                    ],
                }
            )

        return {
            "num_layers": int(self.num_layers),
            "num_levels": int(self.num_levels),
            "decode_tail_chunk_size": int(
                self.decode_tail_chunk_size
            ),
            "actual_compressed_k_storage_bytes": int(
                self.storage_bytes()
            ),
            "layers": layer_reports,
        }