from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch

from contextlib import contextmanager

from turboquant.mse_quant import MSEEncoding
from turboquant.qjl import QJLEncoding
from turboquant.prod_quant import (
    ProdEncoding,
    RuntimePackedProdEncoding,
    turboquant_prod_quantize_3bit,
    turboquant_prod_quantize_3bit_runtime_packed_qjl,
    turboquant_prod_inner_product_estimate,
)
from turboquant.packing import (
    unpack_2bit_indices,
    unpack_sign_bits,
)

@contextmanager
def _nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield

@dataclass
class TurboQuantLayerKeyStorage:
    """
    Preallocated storage for one layer's TurboQuant compressed K cache.

    Physical layout:
      packed_mse_indices_buffer:   [B, H, capacity, D/4]
      mse_norms_buffer:            [B, H, capacity]
      packed_qjl_sign_bits_buffer: [B, H, capacity, M/8]
      qjl_residual_norms_buffer:   [B, H, capacity]

    Optional fp V:
      value_states_buffer:         [B, H, capacity, Dv]

    Only the prefix [:, :, :seq_len, ...] is active.
    """

    packed_mse_indices_buffer: Optional[torch.Tensor] = None
    mse_norms_buffer: Optional[torch.Tensor] = None
    packed_qjl_sign_bits_buffer: Optional[torch.Tensor] = None
    qjl_residual_norms_buffer: Optional[torch.Tensor] = None
    value_states_buffer: Optional[torch.Tensor] = None

    seq_len_value: int = 0
    capacity_value: int = 0

    def is_empty(self) -> bool:
        return self.packed_mse_indices_buffer is None

    def seq_len(self) -> int:
        return int(self.seq_len_value)

    def capacity(self) -> int:
        return int(self.capacity_value)

    # ------------------------------------------------------------------
    # Active views, mostly for Python correctness path and reports
    # ------------------------------------------------------------------

    @property
    def packed_mse_indices(self) -> torch.Tensor:
        if self.packed_mse_indices_buffer is None:
            raise RuntimeError("packed_mse_indices buffer is empty.")
        return self.packed_mse_indices_buffer[:, :, : self.seq_len_value, :]

    @property
    def mse_norms(self) -> torch.Tensor:
        if self.mse_norms_buffer is None:
            raise RuntimeError("mse_norms buffer is empty.")
        return self.mse_norms_buffer[:, :, : self.seq_len_value]

    @property
    def packed_qjl_sign_bits(self) -> torch.Tensor:
        if self.packed_qjl_sign_bits_buffer is None:
            raise RuntimeError("packed_qjl_sign_bits buffer is empty.")
        return self.packed_qjl_sign_bits_buffer[:, :, : self.seq_len_value, :]

    @property
    def qjl_residual_norms(self) -> torch.Tensor:
        if self.qjl_residual_norms_buffer is None:
            raise RuntimeError("qjl_residual_norms buffer is empty.")
        return self.qjl_residual_norms_buffer[:, :, : self.seq_len_value]

    @property
    def value_states(self) -> torch.Tensor:
        if self.value_states_buffer is None:
            raise RuntimeError("value_states buffer is empty.")
        return self.value_states_buffer[:, :, : self.seq_len_value, :]


class TurboQuantKeyCache:
    """
    Preallocated TurboQuant compressed K cache.

    Compared with the older torch.cat implementation:
      - decode append writes in-place into preallocated buffers
      - no recurrent reallocation / concatenation of full K cache
      - CUDA score kernel reads full buffer + seq_len stride directly
    """

    def __init__(
        self,
        num_layers: int,
        rotation: torch.Tensor,
        centroids: torch.Tensor,
        sketch: torch.Tensor,
        max_cache_len: Optional[int] = None,
    ):
        self.num_layers = int(num_layers)
        self.rotation = rotation
        self.centroids = centroids
        self.sketch = sketch
        self.max_cache_len = (
            int(max_cache_len)
            if max_cache_len is not None
            else None
        )

        # Combined query transform:
        # q @ combined_query_transform.T
        # first D dims -> q_rot
        # remaining M dims -> Sq
        self.combined_query_transform = torch.cat(
            [rotation, sketch],
            dim=0,
        ).contiguous()

        self.layers = [
            TurboQuantLayerKeyStorage()
            for _ in range(self.num_layers)
        ]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_seq_length(self, layer_idx: int) -> int:
        return self.layers[layer_idx].seq_len()

    def get_capacity(self, layer_idx: int) -> int:
        return self.layers[layer_idx].capacity()

    def get_num_layers(self) -> int:
        return self.num_layers

    # ------------------------------------------------------------------
    # Allocation / growth
    # ------------------------------------------------------------------

    def _choose_initial_capacity(self, required: int) -> int:
        if self.max_cache_len is not None:
            if required > self.max_cache_len:
                raise RuntimeError(
                    f"required cache len {required} exceeds max_cache_len={self.max_cache_len}"
                )
            return self.max_cache_len

        return max(required, 16)

    def _choose_grown_capacity(self, current: int, required: int) -> int:
        if self.max_cache_len is not None:
            if required > self.max_cache_len:
                raise RuntimeError(
                    f"required cache len {required} exceeds max_cache_len={self.max_cache_len}"
                )
            return self.max_cache_len

        return max(required, max(16, current * 2))

    @torch.no_grad()
    def _allocate_empty_layer(
        self,
        layer: TurboQuantLayerKeyStorage,
        packed_mse_indices: torch.Tensor,
        mse_norms: torch.Tensor,
        packed_qjl_sign_bits: torch.Tensor,
        qjl_residual_norms: torch.Tensor,
        value_states: Optional[torch.Tensor],
        capacity: int,
    ) -> None:
        B, H, _, packed_D = packed_mse_indices.shape
        packed_M = packed_qjl_sign_bits.shape[-1]

        device = packed_mse_indices.device

        layer.packed_mse_indices_buffer = torch.empty(
            (B, H, capacity, packed_D),
            dtype=packed_mse_indices.dtype,
            device=device,
        )

        layer.mse_norms_buffer = torch.empty(
            (B, H, capacity),
            dtype=mse_norms.dtype,
            device=device,
        )

        layer.packed_qjl_sign_bits_buffer = torch.empty(
            (B, H, capacity, packed_M),
            dtype=packed_qjl_sign_bits.dtype,
            device=device,
        )

        layer.qjl_residual_norms_buffer = torch.empty(
            (B, H, capacity),
            dtype=qjl_residual_norms.dtype,
            device=device,
        )

        if value_states is not None:
            Dv = value_states.shape[-1]
            layer.value_states_buffer = torch.empty(
                (B, H, capacity, Dv),
                dtype=value_states.dtype,
                device=value_states.device,
            )

        layer.seq_len_value = 0
        layer.capacity_value = int(capacity)

    @torch.no_grad()
    def _grow_existing_layer(
        self,
        layer: TurboQuantLayerKeyStorage,
        new_capacity: int,
    ) -> None:
        old_len = layer.seq_len_value
        old_capacity = layer.capacity_value

        if new_capacity <= old_capacity:
            return

        def grow_4d(buf: torch.Tensor) -> torch.Tensor:
            B, H, _, C = buf.shape
            out = torch.empty(
                (B, H, new_capacity, C),
                dtype=buf.dtype,
                device=buf.device,
            )
            if old_len > 0:
                out[:, :, :old_len, :].copy_(buf[:, :, :old_len, :])
            return out

        def grow_3d(buf: torch.Tensor) -> torch.Tensor:
            B, H, _ = buf.shape
            out = torch.empty(
                (B, H, new_capacity),
                dtype=buf.dtype,
                device=buf.device,
            )
            if old_len > 0:
                out[:, :, :old_len].copy_(buf[:, :, :old_len])
            return out

        layer.packed_mse_indices_buffer = grow_4d(
            layer.packed_mse_indices_buffer
        )
        layer.mse_norms_buffer = grow_3d(
            layer.mse_norms_buffer
        )
        layer.packed_qjl_sign_bits_buffer = grow_4d(
            layer.packed_qjl_sign_bits_buffer
        )
        layer.qjl_residual_norms_buffer = grow_3d(
            layer.qjl_residual_norms_buffer
        )

        if layer.value_states_buffer is not None:
            layer.value_states_buffer = grow_4d(
                layer.value_states_buffer
            )

        layer.capacity_value = int(new_capacity)

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    @torch.no_grad()
    def append(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Append new key states into preallocated compressed buffers.

        key_states:
          [B, H, T_new, D]

        value_states:
          optional [B, H, T_new, Dv]
        """
        if key_states.ndim != 4:
            raise ValueError(
                f"key_states must be [B,H,T,D], got shape={tuple(key_states.shape)}"
            )

        B, H, T_new, D = key_states.shape

        if D != self.rotation.shape[0]:
            raise ValueError(
                f"head_dim mismatch: key D={D}, rotation dim={self.rotation.shape[0]}"
            )

        layer = self.layers[layer_idx]

        current_len = layer.seq_len()
        required_len = current_len + T_new

        key_flat = key_states.float().reshape(-1, D)

        with _nvtx_range("tq_append_quantize"):
            enc: RuntimePackedProdEncoding = (
                turboquant_prod_quantize_3bit_runtime_packed_qjl(
                    x=key_flat,
                    rotation=self.rotation,
                    centroids=self.centroids,
                    sketch=self.sketch,
                )
            )

        M = self.sketch.shape[0]

        packed_mse_indices = enc.packed_mse_indices.reshape(
            B,
            H,
            T_new,
            D // 4,
        )

        mse_norms = enc.mse_norms.reshape(
            B,
            H,
            T_new,
        )

        packed_qjl_sign_bits = enc.packed_qjl_sign_bits.reshape(
            B,
            H,
            T_new,
            M // 8,
        )

        qjl_residual_norms = enc.qjl_residual_norms.reshape(
            B,
            H,
            T_new,
        )
        
        if layer.is_empty():
            capacity = self._choose_initial_capacity(required_len)

            self._allocate_empty_layer(
                layer=layer,
                packed_mse_indices=packed_mse_indices,
                mse_norms=mse_norms,
                packed_qjl_sign_bits=packed_qjl_sign_bits,
                qjl_residual_norms=qjl_residual_norms,
                value_states=value_states,
                capacity=capacity,
            )

        elif required_len > layer.capacity():
            new_capacity = self._choose_grown_capacity(
                current=layer.capacity(),
                required=required_len,
            )
            self._grow_existing_layer(
                layer=layer,
                new_capacity=new_capacity,
            )

        write_begin = current_len
        write_end = required_len

        with _nvtx_range("tq_append_write"):
            layer.packed_mse_indices_buffer[
                :, :, write_begin:write_end, :
            ].copy_(packed_mse_indices)

            layer.mse_norms_buffer[
                :, :, write_begin:write_end
            ].copy_(mse_norms)

            layer.packed_qjl_sign_bits_buffer[
                :, :, write_begin:write_end, :
            ].copy_(packed_qjl_sign_bits)

            layer.qjl_residual_norms_buffer[
                :, :, write_begin:write_end
            ].copy_(qjl_residual_norms)

            if value_states is not None:
                if layer.value_states_buffer is None:
                    Dv = value_states.shape[-1]
                    layer.value_states_buffer = torch.empty(
                        (
                            B,
                            H,
                            layer.capacity(),
                            Dv,
                        ),
                        dtype=value_states.dtype,
                        device=value_states.device,
                    )

                layer.value_states_buffer[
                    :, :, write_begin:write_end, :
                ].copy_(value_states)

        layer.seq_len_value = int(required_len)

    # ------------------------------------------------------------------
    # Rebuild Python ProdEncoding from active compressed prefix
    # ------------------------------------------------------------------

    def _layer_prod_encoding_flat(
        self,
        layer_idx: int,
    ) -> ProdEncoding:
        layer = self.layers[layer_idx]

        if layer.is_empty():
            raise RuntimeError(f"Layer {layer_idx} cache is empty.")

        B, H, T, _ = layer.packed_mse_indices.shape

        D = int(self.rotation.shape[0])
        M = int(self.sketch.shape[0])

        mse_indices = unpack_2bit_indices(
            layer.packed_mse_indices,
            original_dim=D,
        )

        # Current QJLEncoding expects packed 1-bit signs, not unpacked {-1,+1}
        # tensors. Keep the cache payload packed and let qjl_inner_product_estimate()
        # perform the unpacking, matching the current QJL API.
        packed_qjl_sign_bits = layer.packed_qjl_sign_bits.reshape(
            B * H * T,
            M // 8,
        ).contiguous()

        return ProdEncoding(
            mse=MSEEncoding(
                indices=mse_indices.reshape(B * H * T, D),
                norms=layer.mse_norms.reshape(B * H * T),
            ),
            qjl_residual=QJLEncoding(
                packed_sign_bits=packed_qjl_sign_bits,
                norms=layer.qjl_residual_norms.reshape(B * H * T),
            ),
        )

    # ------------------------------------------------------------------
    # Python reference score path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Python correctness path.
        Returns [B,H,Q,T].
        """
        if query_states.ndim != 4:
            raise ValueError(
                f"query_states must be [B,H,Q,D], got shape={tuple(query_states.shape)}"
            )

        layer = self.layers[layer_idx]
        if layer.is_empty():
            raise RuntimeError(f"Layer {layer_idx} cache is empty.")

        Bq, Hq, Q, Dq = query_states.shape
        Bk, Hk, T, _ = layer.packed_mse_indices.shape
        Dk = int(self.rotation.shape[0])

        if (Bq, Hq, Dq) != (Bk, Hk, Dk):
            raise ValueError(
                "query/cache shape mismatch: "
                f"query=[{Bq},{Hq},{Q},{Dq}], "
                f"cache=[{Bk},{Hk},{T},{Dk}]"
            )

        enc = self._layer_prod_encoding_flat(layer_idx)

        score_chunks = []

        for q_idx in range(Q):
            q_one = query_states[:, :, q_idx:q_idx + 1, :]

            q_repeat = (
                q_one.float()
                .reshape(Bq * Hq, 1, Dq)
                .expand(-1, T, -1)
                .reshape(Bq * Hq * T, Dq)
            )

            est_dot = turboquant_prod_inner_product_estimate(
                q=q_repeat,
                encoding=enc,
                rotation=self.rotation,
                centroids=self.centroids,
                sketch=self.sketch,
            )

            score_one = est_dot.reshape(Bq, Hq, 1, T)
            score_chunks.append(score_one)

        return torch.cat(score_chunks, dim=2)

    # ------------------------------------------------------------------
    # V access
    # ------------------------------------------------------------------

    def get_value_states(
        self,
        layer_idx: int,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        return layer.value_states

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def actual_storage_bytes(
        self,
        layer_idx: Optional[int] = None,
    ) -> int:
        """
        Active compressed K payload only.
        Excludes inactive capacity tail.
        """
        if layer_idx is None:
            return sum(
                self.actual_storage_bytes(i)
                for i in range(self.num_layers)
            )

        layer = self.layers[layer_idx]
        if layer.is_empty():
            return 0

        B, H, _, packed_D = layer.packed_mse_indices_buffer.shape
        packed_M = layer.packed_qjl_sign_bits_buffer.shape[-1]
        T = layer.seq_len()

        total = 0

        total += (
            B * H * T * packed_D
            * layer.packed_mse_indices_buffer.element_size()
        )
        total += (
            B * H * T
            * layer.mse_norms_buffer.element_size()
        )
        total += (
            B * H * T * packed_M
            * layer.packed_qjl_sign_bits_buffer.element_size()
        )
        total += (
            B * H * T
            * layer.qjl_residual_norms_buffer.element_size()
        )

        return int(total)

    def allocated_storage_bytes(
        self,
        layer_idx: Optional[int] = None,
    ) -> int:
        """
        Physical compressed K buffer allocation.
        Includes inactive tail capacity.
        """
        if layer_idx is None:
            return sum(
                self.allocated_storage_bytes(i)
                for i in range(self.num_layers)
            )

        layer = self.layers[layer_idx]
        if layer.is_empty():
            return 0

        tensors = [
            layer.packed_mse_indices_buffer,
            layer.mse_norms_buffer,
            layer.packed_qjl_sign_bits_buffer,
            layer.qjl_residual_norms_buffer,
        ]

        total = 0
        for t in tensors:
            total += t.numel() * t.element_size()

        return int(total)

    def value_actual_storage_bytes(
        self,
        layer_idx: Optional[int] = None,
    ) -> int:
        if layer_idx is None:
            return sum(
                self.value_actual_storage_bytes(i)
                for i in range(self.num_layers)
            )

        layer = self.layers[layer_idx]
        if layer.value_states_buffer is None:
            return 0

        B, H, _, Dv = layer.value_states_buffer.shape
        T = layer.seq_len()

        return int(
            B * H * T * Dv
            * layer.value_states_buffer.element_size()
        )

    def value_allocated_storage_bytes(
        self,
        layer_idx: Optional[int] = None,
    ) -> int:
        if layer_idx is None:
            return sum(
                self.value_allocated_storage_bytes(i)
                for i in range(self.num_layers)
            )

        layer = self.layers[layer_idx]
        if layer.value_states_buffer is None:
            return 0

        return int(
            layer.value_states_buffer.numel()
            * layer.value_states_buffer.element_size()
        )

    def target_packed_k_bytes(
        self,
        layer_idx: Optional[int] = None,
    ) -> int:
        if layer_idx is None:
            return sum(
                self.target_packed_k_bytes(i)
                for i in range(self.num_layers)
            )

        layer = self.layers[layer_idx]
        if layer.is_empty():
            return 0

        B, H, _, packed_D = layer.packed_mse_indices_buffer.shape
        packed_M = layer.packed_qjl_sign_bits_buffer.shape[-1]
        T = layer.seq_len()

        D = packed_D * 4
        M = packed_M * 8

        num_vectors = B * H * T

        bits_indices = num_vectors * D * 2
        bits_signs = num_vectors * M * 1
        bits_norms = num_vectors * 32
        bits_residual_norms = num_vectors * 32

        total_bits = (
            bits_indices
            + bits_signs
            + bits_norms
            + bits_residual_norms
        )

        return int((total_bits + 7) // 8)

    def report(
        self,
        layer_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        if layer_idx is None:
            actual = self.actual_storage_bytes()
            allocated = self.allocated_storage_bytes()
            target = self.target_packed_k_bytes()

            return {
                "num_layers": self.num_layers,
                "actual_storage_bytes": actual,
                "allocated_storage_bytes": allocated,
                "target_packed_k_bytes": target,
                "actual_over_target_ratio": (
                    float(actual) / float(target)
                    if target > 0 else None
                ),
                "allocated_over_active_ratio": (
                    float(allocated) / float(actual)
                    if actual > 0 else None
                ),
                "layers": [
                    self.report(i)
                    for i in range(self.num_layers)
                    if self.get_seq_length(i) > 0
                ],
            }

        layer = self.layers[layer_idx]

        actual = self.actual_storage_bytes(layer_idx)
        allocated = self.allocated_storage_bytes(layer_idx)
        target = self.target_packed_k_bytes(layer_idx)

        active_mse_shape = None
        active_sign_shape = None
        allocated_mse_shape = None
        allocated_sign_shape = None

        if not layer.is_empty():
            active_mse_shape = list(layer.packed_mse_indices.shape)
            active_sign_shape = list(layer.packed_qjl_sign_bits.shape)
            allocated_mse_shape = list(layer.packed_mse_indices_buffer.shape)
            allocated_sign_shape = list(layer.packed_qjl_sign_bits_buffer.shape)

        return {
            "layer_idx": layer_idx,
            "seq_len": layer.seq_len(),
            "capacity": layer.capacity(),
            "actual_storage_bytes": actual,
            "allocated_storage_bytes": allocated,
            "target_packed_k_bytes": target,
            "actual_over_target_ratio": (
                float(actual) / float(target)
                if target > 0 else None
            ),
            "allocated_over_active_ratio": (
                float(allocated) / float(actual)
                if actual > 0 else None
            ),
            "packed_mse_indices_shape": active_mse_shape,
            "packed_qjl_sign_bits_shape": active_sign_shape,
            "allocated_packed_mse_indices_shape": allocated_mse_shape,
            "allocated_packed_qjl_sign_bits_shape": allocated_sign_shape,
        }