from __future__ import annotations

from typing import Optional, Dict, Any

import torch

from transformers.cache_utils import Cache
from turboquant.key_cache import TurboQuantKeyCache


class TurboQuantRuntimeCache(Cache):
    """
    Runtime cache for:
      - compressed packed TurboQuant K
      - full-precision V

    K and V are both owned by TurboQuantKeyCache storage so that:
      - both use the same seq_len
      - both support preallocated in-place append
    """

    def __init__(
        self,
        num_layers: int,
        rotation: torch.Tensor,
        centroids: torch.Tensor,
        sketch: torch.Tensor,
        max_cache_len: Optional[int] = None,
    ):
        super().__init__()
        
        self.num_layers = int(num_layers)
        self.max_cache_len = (
            int(max_cache_len)
            if max_cache_len is not None
            else None
        )

        self.tq_key_cache = TurboQuantKeyCache(
            num_layers=num_layers,
            rotation=rotation,
            centroids=centroids,
            sketch=sketch,
            max_cache_len=max_cache_len,
        )

        self._seen_tokens = 0
        # Legacy tuple-like compatibility for Hugging Face generate().
        # Some transformers versions still call:
        #   past_key_values[0][0].shape[2]
        # before the first forward pass.
        #
        # These empty tensors are only metadata placeholders for the
        # "empty cache" state; they are not used for attention computation.
        head_dim = int(rotation.shape[0])
        self._legacy_empty_key = torch.empty(
            (1, 1, 0, head_dim),
            dtype=torch.float16,
            device=rotation.device,
        )
        self._legacy_empty_value = torch.empty(
            (1, 1, 0, head_dim),
            dtype=torch.float16,
            device=rotation.device,
        )

    # ------------------------------------------------------------------
    # Hugging Face cache-like interface
    # ------------------------------------------------------------------

    @property
    def seen_tokens(self) -> int:
        return int(self._seen_tokens)

    def __len__(self) -> int:
        return self.num_layers
    
    def __getitem__(self, layer_idx: int):
        """
        Legacy tuple-like compatibility for Hugging Face generation code.

        Transformers' LLaMA generation path may still inspect:
            past_key_values[0][0].shape[2]

        Our actual K cache is compressed and cannot be returned as a dense
        historical K tensor. For this compatibility path, only the sequence
        length metadata is needed, so:

          - empty cache:
              return zero-length placeholder tensors
          - non-empty cache:
              return active V cache twice, because it has shape [B,H,T,D]
              and therefore exposes the correct T at shape[2]

        Attention computation does NOT use this dense "key-like" tensor;
        patched attention reads compressed K from tq_key_cache directly.
        """
        layer = self.tq_key_cache.layers[layer_idx]

        if layer.value_states_buffer is None:
            return (
                self._legacy_empty_key,
                self._legacy_empty_value,
            )

        active_value_states = layer.value_states

        return (
            active_value_states,
            active_value_states,
        )

    def get_seq_length(
        self,
        layer_idx: int = 0,
    ) -> int:
        return self.tq_key_cache.get_seq_length(layer_idx)

    def get_max_length(self) -> Optional[int]:
        return self.max_cache_len

    def get_usable_length(
        self,
        new_seq_length: int,
        layer_idx: int = 0,
    ) -> int:
        return self.get_seq_length(layer_idx)

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ):
        """
        Append one layer's K/V into preallocated buffers.

        Return:
          (key_states, full_active_value_cache)

        The patched attention path does not rely on dense historical K;
        it computes scores from self.tq_key_cache directly.
        Returning key_states keeps the call signature cache-compatible.
        """
        self.tq_key_cache.append(
            layer_idx=layer_idx,
            key_states=key_states,
            value_states=value_states,
        )

        if layer_idx == 0:
            self._seen_tokens = self.get_seq_length(layer_idx)

        full_value_cache = self.tq_key_cache.get_value_states(layer_idx)

        return key_states, full_value_cache

    @torch.no_grad()
    def reorder_cache(
        self,
        beam_idx: torch.LongTensor,
    ) -> None:
        """
        Minimal beam-search compatibility.
        Greedy benchmark does not use this, but it is cheap to preserve.
        """
        for layer in self.tq_key_cache.layers:
            if layer.is_empty():
                continue

            layer.packed_mse_indices_buffer = (
                layer.packed_mse_indices_buffer.index_select(0, beam_idx)
            )
            layer.mse_norms_buffer = (
                layer.mse_norms_buffer.index_select(0, beam_idx)
            )
            layer.packed_qjl_sign_bits_buffer = (
                layer.packed_qjl_sign_bits_buffer.index_select(0, beam_idx)
            )
            layer.qjl_residual_norms_buffer = (
                layer.qjl_residual_norms_buffer.index_select(0, beam_idx)
            )

            if layer.value_states_buffer is not None:
                layer.value_states_buffer = (
                    layer.value_states_buffer.index_select(0, beam_idx)
                )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        compressed_k_report = self.tq_key_cache.report()

        fp_value_cache_layers = []

        for layer_idx, layer in enumerate(self.tq_key_cache.layers):
            if layer.value_states_buffer is None:
                continue

            active_value = layer.value_states

            fp_value_cache_layers.append(
                {
                    "layer_idx": layer_idx,
                    "shape": list(active_value.shape),
                    "capacity": int(layer.capacity()),
                    "bytes": self.tq_key_cache.value_actual_storage_bytes(layer_idx),
                    "allocated_bytes": self.tq_key_cache.value_allocated_storage_bytes(layer_idx),
                }
            )

        return {
            "seen_tokens": self.seen_tokens,
            "compressed_k": compressed_k_report,
            "fp_value_cache_bytes": self.tq_key_cache.value_actual_storage_bytes(),
            "fp_value_cache_allocated_bytes": self.tq_key_cache.value_allocated_storage_bytes(),
            "fp_value_cache_layers": fp_value_cache_layers,
        }