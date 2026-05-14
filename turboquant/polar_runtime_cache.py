from __future__ import annotations

from typing import Any

import torch

from turboquant.polar_key_cache import (
    PolarProdKeyCache,
)
from turboquant.polarquant_quant import (
    PolarAngleCodebooks,
)
from transformers.cache_utils import Cache

class TurboQuantPolarRuntimeCache(Cache):
    """
    Runtime cache wrapper for the real TurboQuant path:

      K cache:
        Polar Stage-1 + QJL residual
        stored by PolarProdKeyCache

      V cache:
        dense fp16 / model dtype tensor
        stored layer-wise as ordinary concatenated value states

    This class is the bridge before wiring into llama_score_patch.py.

    Current supported workflow:
      - update(...)
      - score(...)
      - get_value_states(...)
      - get_seq_length(...)
      - report(...)

    The next step will make llama_score_patch.py recognize this cache.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        codebooks: PolarAngleCodebooks,
        sketch: torch.Tensor,
        num_levels: int = 4,
    ) -> None:
        super().__init__()
        
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.num_layers = int(num_layers)
        self.codebooks = codebooks
        self.sketch = sketch
        self.num_levels = int(num_levels)

        self.key_cache = PolarProdKeyCache(
            num_layers=self.num_layers,
            codebooks=self.codebooks,
            sketch=self.sketch,
            num_levels=self.num_levels,
        )

        self.value_cache: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]

        self._seen_tokens = 0

    @property
    def seen_tokens(self) -> int:
        return int(self._seen_tokens)

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
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runtime-style cache update.

        Args:
            key_states:
                [B,H,T_new,D]

            value_states:
                [B,H,T_new,D]

            layer_idx:
                transformer layer index

            cache_kwargs:
                kept for HuggingFace-style compatibility;
                currently unused.

        Returns:
            key_states, full_value_states

        Note:
            We return the original key_states only as a compatibility value.
            The compressed K representation is kept internally in self.key_cache.
            The upcoming llama_score_patch integration will compute scores from
            self.key_cache.score(...), not from this returned key_states.
        """
        del cache_kwargs

        self._check_layer_idx(layer_idx)

        if key_states.ndim != 4:
            raise ValueError(
                f"key_states must be [B,H,T,D], got {tuple(key_states.shape)}"
            )

        if value_states.ndim != 4:
            raise ValueError(
                f"value_states must be [B,H,T,D], got {tuple(value_states.shape)}"
            )

        if tuple(key_states.shape[:3]) != tuple(value_states.shape[:3]):
            raise ValueError(
                "key_states and value_states must match in [B,H,T]. "
                f"Got K={tuple(key_states.shape)}, V={tuple(value_states.shape)}"
            )

        # ------------------------------------------------------------
        # 1. Append compressed K into PolarProdKeyCache
        # ------------------------------------------------------------
        self.key_cache.append(
            layer_idx=layer_idx,
            key_states=key_states,
        )

        # ------------------------------------------------------------
        # 2. Append dense V cache
        # ------------------------------------------------------------
        existing_v = self.value_cache[layer_idx]

        if existing_v is None:
            full_v = value_states.contiguous()
        else:
            full_v = torch.cat(
                [existing_v, value_states],
                dim=2,
            ).contiguous()

        self.value_cache[layer_idx] = full_v

        # ------------------------------------------------------------
        # 3. Update seen_tokens.
        # Follow the conventional cache pattern:
        # only layer 0 advances global sequence length.
        # ------------------------------------------------------------
        if layer_idx == 0:
            self._seen_tokens += int(key_states.shape[2])

        # key_states returned only for interface compatibility;
        # full_v is the dense V cache attention needs.
        return key_states, full_v

    @torch.no_grad()
    def score(
        self,
        *,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute compressed K-cache attention scores.

        Args:
            query_states:
                [B,H,1,D]

        Returns:
            [B,H,1,T_cache]
        """
        self._check_layer_idx(layer_idx)

        return self.key_cache.score(
            layer_idx=layer_idx,
            query_states=query_states,
        )

    def get_value_states(
        self,
        *,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Return dense V cache for a layer:
          [B,H,T_cache,D]
        """
        self._check_layer_idx(layer_idx)

        value_states = self.value_cache[layer_idx]

        if value_states is None:
            raise RuntimeError(
                f"Value cache for layer {layer_idx} is empty."
            )

        return value_states

    def get_seq_length(
        self,
        layer_idx: int = 0,
    ) -> int:
        """
        HuggingFace-style cache length query.

        For this cache, all layers should have the same sequence length.
        """
        self._check_layer_idx(layer_idx)
        return self.key_cache.seq_len(
            layer_idx=layer_idx,
        )

    def get_max_length(self) -> None:
        """
        Compatibility stub.
        This dynamic cache does not impose a static max length.
        """
        return None

    def __len__(self) -> int:
        return int(self.num_layers)

    def report(
        self,
    ) -> dict[str, Any]:
        value_layer_reports: list[dict[str, Any]] = []

        total_value_cache_bytes = 0

        for layer_idx, value_states in enumerate(self.value_cache):
            if value_states is None:
                value_layer_reports.append(
                    {
                        "layer_idx": int(layer_idx),
                        "shape": None,
                        "bytes": 0,
                    }
                )
                continue

            num_bytes = (
                value_states.numel() *
                value_states.element_size()
            )

            total_value_cache_bytes += int(num_bytes)

            value_layer_reports.append(
                {
                    "layer_idx": int(layer_idx),
                    "shape": [
                        int(s) for s in value_states.shape
                    ],
                    "bytes": int(num_bytes),
                }
            )

        compressed_k_report = self.key_cache.report()

        return {
            "num_layers": int(self.num_layers),
            "num_levels": int(self.num_levels),
            "seen_tokens": int(self.seen_tokens),
            "compressed_k": compressed_k_report,
            "compressed_k_storage_bytes": int(
                compressed_k_report["actual_compressed_k_storage_bytes"]
            ),
            "fp_value_cache_bytes": int(total_value_cache_bytes),
            "fp_value_cache_layers": value_layer_reports,
        }
