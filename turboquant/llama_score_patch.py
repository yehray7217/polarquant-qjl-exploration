from __future__ import annotations

import math
import os
import types
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from turboquant.key_cache import TurboQuantKeyCache
from turboquant.runtime_cache import TurboQuantRuntimeCache
from turboquant.cuda_score import (
    turboquant_decode_score_cuda_from_cache,
)


# ============================================================
# NVTX
# ============================================================

# Default ON for the current profiling phase.
# Later, for non-profile benchmark, you can disable with:
#   TQ_INNER_NVTX=0 python ...
_ENABLE_TQ_INNER_NVTX = os.environ.get("TQ_INNER_NVTX", "1") != "0"


@contextmanager
def _nvtx_range(name: str):
    if _ENABLE_TQ_INNER_NVTX and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


# ============================================================
# TurboQuant patch state
# ============================================================

class TurboQuantScoreState:
    """
    Shared state used by all patched LLaMA attention modules.

    Contains:
      - fixed rotation matrix
      - centroids
      - QJL sketch
      - optional side key cache
      - runtime debugging switches
    """

    def __init__(
        self,
        head_dim: int,
        qjl_m: int = 256,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.float32,
        rotation_seed: int = 123,
        sketch_seed: int = 456,
        use_cuda_decode_score: bool = True,
        compare_cuda_decode_score: bool = False,
    ):
        from turboquant.mse_quant import (
            make_random_rotation,
            get_2bit_centroids,
        )
        from turboquant.qjl import make_gaussian_sketch

        self.head_dim = int(head_dim)
        self.qjl_m = int(qjl_m)
        self.device = torch.device(device)
        self.dtype = dtype

        self.rotation = make_random_rotation(
            d=head_dim,
            device=device,
            dtype=dtype,
            seed=rotation_seed,
        )

        self.centroids = get_2bit_centroids(
            d=head_dim,
            device=device,
            dtype=dtype,
        )

        self.sketch = make_gaussian_sketch(
            d=head_dim,
            m=qjl_m,
            device=device,
            dtype=dtype,
            seed=sketch_seed,
        )

        # Used by the older "side-cache" path, where the model still keeps
        # normal dense KV cache for generation compatibility while attention
        # scores are estimated from TurboQuant K.
        self.key_cache: Optional[TurboQuantKeyCache] = None

        self.use_cuda_decode_score = bool(use_cuda_decode_score)
        self.compare_cuda_decode_score = bool(compare_cuda_decode_score)
        self.decode_score_diagnostics: list[dict[str, Any]] = []

    def ensure_side_key_cache(self, num_layers: int) -> TurboQuantKeyCache:
        if self.key_cache is None:
            self.key_cache = TurboQuantKeyCache(
                num_layers=num_layers,
                rotation=self.rotation,
                centroids=self.centroids,
                sketch=self.sketch,
            )
        return self.key_cache


# ============================================================
# Compatibility helpers
# ============================================================

def _get_layer_idx(attn_module) -> int:
    if getattr(attn_module, "layer_idx", None) is None:
        raise RuntimeError(
            "Patched LlamaAttention requires self.layer_idx. "
            "This transformers version should provide it."
        )
    return int(attn_module.layer_idx)


def _rotary_emb_compat(
    attn_module,
    value_states: torch.Tensor,
    kv_seq_len: int,
):
    """
    Compatible with the transformers version used in this repo:
      rotary_emb(value_states, seq_len=kv_seq_len)

    A fallback branch is kept for minor API differences.
    """
    try:
        cos, sin = attn_module.rotary_emb(
            value_states,
            seq_len=kv_seq_len,
        )
    except TypeError:
        cos, sin = attn_module.rotary_emb(
            value_states,
            kv_seq_len,
        )

    return cos, sin


def _project_qkv(
    attn_module,
    hidden_states: torch.Tensor,
):
    """
    Mirrors LLaMA q/k/v projection behavior, including pretraining_tp>1.
    For Llama-2-7B in this project pretraining_tp is normally 1,
    but preserving this branch keeps the patch safer.
    """
    bsz, q_len, _ = hidden_states.size()

    pretraining_tp = getattr(attn_module.config, "pretraining_tp", 1)

    if pretraining_tp > 1:
        key_value_slicing = (
            attn_module.num_key_value_heads * attn_module.head_dim
        ) // pretraining_tp

        query_slices = attn_module.q_proj.weight.split(
            (attn_module.num_heads * attn_module.head_dim) // pretraining_tp,
            dim=0,
        )
        key_slices = attn_module.k_proj.weight.split(
            key_value_slicing,
            dim=0,
        )
        value_slices = attn_module.v_proj.weight.split(
            key_value_slicing,
            dim=0,
        )

        query_states = [
            nn.functional.linear(hidden_states, query_slices[i])
            for i in range(pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            nn.functional.linear(hidden_states, key_slices[i])
            for i in range(pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            nn.functional.linear(hidden_states, value_slices[i])
            for i in range(pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = attn_module.q_proj(hidden_states)
        key_states = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)

    query_states = query_states.view(
        bsz,
        q_len,
        attn_module.num_heads,
        attn_module.head_dim,
    ).transpose(1, 2)

    key_states = key_states.view(
        bsz,
        q_len,
        attn_module.num_key_value_heads,
        attn_module.head_dim,
    ).transpose(1, 2)

    value_states = value_states.view(
        bsz,
        q_len,
        attn_module.num_key_value_heads,
        attn_module.head_dim,
    ).transpose(1, 2)

    return query_states, key_states, value_states


def _apply_output_projection(
    attn_module,
    attn_output: torch.Tensor,
):
    """
    Mirrors LLaMA o_proj, including pretraining_tp>1.
    """
    bsz, _, q_len, _ = attn_output.shape

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(
        bsz,
        q_len,
        attn_module.hidden_size,
    )

    pretraining_tp = getattr(attn_module.config, "pretraining_tp", 1)

    if pretraining_tp > 1:
        attn_output = attn_output.split(
            attn_module.hidden_size // pretraining_tp,
            dim=2,
        )
        o_proj_slices = attn_module.o_proj.weight.split(
            attn_module.hidden_size // pretraining_tp,
            dim=1,
        )
        attn_output = sum(
            [
                nn.functional.linear(attn_output[i], o_proj_slices[i])
                for i in range(pretraining_tp)
            ]
        )
    else:
        attn_output = attn_module.o_proj(attn_output)

    return attn_output


def _validate_attention_mask_shape(
    attention_mask: torch.Tensor,
    bsz: int,
    q_len: int,
    kv_seq_len: int,
):
    expected_shape = (bsz, 1, q_len, kv_seq_len)

    if attention_mask.size() != expected_shape:
        raise ValueError(
            f"Attention mask should be of size {expected_shape}, "
            f"but is {tuple(attention_mask.size())}"
        )


# ============================================================
# Side-cache TurboQuant score path
# ============================================================

@torch.no_grad()
def _turboquant_scores_from_key_cache(
    layer_idx: int,
    query_states: torch.Tensor,
    tq_state: TurboQuantScoreState,
) -> torch.Tensor:
    """
    Python reference score path for the older side-cache mode.
    """
    if tq_state.key_cache is None:
        raise RuntimeError(
            "TurboQuant side key cache is not initialized."
        )

    return tq_state.key_cache.score(
        layer_idx=layer_idx,
        query_states=query_states,
    )


# ============================================================
# Runtime-cache score computation
# ============================================================

@torch.no_grad()
def _runtime_cache_scores(
    layer_idx: int,
    query_states: torch.Tensor,
    runtime_cache: TurboQuantRuntimeCache,
    tq_state: TurboQuantScoreState,
) -> torch.Tensor:
    """
    Score path for TurboQuantRuntimeCache.

    Decode:
      query_len == 1
      -> CUDA path if enabled

    Prefill / multi-query:
      -> Python reference score path
    """
    tq_key_cache = runtime_cache.tq_key_cache

    query_len = int(query_states.shape[2])

    if query_len == 1:
        score_python = None
        score_cuda = None

        if tq_state.compare_cuda_decode_score:
            score_python = tq_key_cache.score(
                layer_idx=layer_idx,
                query_states=query_states,
            )

        if tq_state.use_cuda_decode_score or tq_state.compare_cuda_decode_score:
            with _nvtx_range("tq_score_cuda"):
                score_cuda = turboquant_decode_score_cuda_from_cache(
                    query_states=query_states,
                    cache=tq_key_cache,
                    layer_idx=layer_idx,
                )

        if tq_state.compare_cuda_decode_score:
            if score_python is None or score_cuda is None:
                raise RuntimeError(
                    "compare_cuda_decode_score=True requires both Python and CUDA scores."
                )

            diff = torch.abs(score_python - score_cuda)

            tq_state.decode_score_diagnostics.append(
                {
                    "layer_idx": int(layer_idx),
                    "seq_len": int(tq_key_cache.get_seq_length(layer_idx)),
                    "score_shape": list(score_python.shape),
                    "max_abs_diff": float(diff.max().item()),
                    "mean_abs_diff": float(diff.mean().item()),
                    "python_abs_max": float(score_python.abs().max().item()),
                    "cuda_abs_max": float(score_cuda.abs().max().item()),
                }
            )

        if tq_state.use_cuda_decode_score:
            if score_cuda is None:
                raise RuntimeError(
                    "CUDA decode score was expected but not computed."
                )
            return score_cuda

        if score_python is not None:
            return score_python

        return tq_key_cache.score(
            layer_idx=layer_idx,
            query_states=query_states,
        )

    # Prefill / non-decode fallback.
    return tq_key_cache.score(
        layer_idx=layer_idx,
        query_states=query_states,
    )


# ============================================================
# Patched attention forward
# ============================================================

def _turboquant_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """
    LLaMA attention forward patched to use TurboQuant K-score estimation.

    Supported modes:
      1. TurboQuantRuntimeCache
         - compressed K
         - fp16 V
         - CUDA decode score for q_len=1

      2. Side-cache fallback
         - dense HF cache still used for values
         - TurboQuant side K cache used for attention scores
         - useful for correctness tests / older smoke tests
    """
    tq_state: TurboQuantScoreState = self._turboquant_score_state
    layer_idx = _get_layer_idx(self)

    bsz, q_len, _ = hidden_states.size()

    # ------------------------------------------------------------
    # Q/K/V projections
    # ------------------------------------------------------------
    query_states, key_states, value_states = _project_qkv(
        self,
        hidden_states,
    )

    # ------------------------------------------------------------
    # Effective KV length for RoPE
    # ------------------------------------------------------------
    kv_seq_len = key_states.shape[-2]

    if past_key_value is not None:
        if hasattr(past_key_value, "get_usable_length"):
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len,
                layer_idx,
            )
        elif isinstance(past_key_value, tuple):
            kv_seq_len += past_key_value[0].shape[-2]

    # ------------------------------------------------------------
    # RoPE
    # ------------------------------------------------------------
    cos, sin = _rotary_emb_compat(
        self,
        value_states=value_states,
        kv_seq_len=int(kv_seq_len),
    )

    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        position_ids,
    )

    # ------------------------------------------------------------
    # Cache update and value-cache construction
    # ------------------------------------------------------------
    present_key_value = past_key_value if use_cache else None

    if isinstance(past_key_value, TurboQuantRuntimeCache):
        # Runtime compressed cache path.
        # update() appends the new K/V into preallocated buffers.
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
        }

        _, value_states = past_key_value.update(
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            cache_kwargs=cache_kwargs,
        )

        present_key_value = past_key_value if use_cache else None

        # In this project Llama-2-7B has H_kv == H_q == 32.
        # repeat_kv is left here for structural compatibility.
        value_states = repeat_kv(
            value_states,
            self.num_key_value_groups,
        )

        attn_weights = _runtime_cache_scores(
            layer_idx=layer_idx,
            query_states=query_states,
            runtime_cache=past_key_value,
            tq_state=tq_state,
        )

    else:
        # --------------------------------------------------------
        # Side-cache / legacy fallback path
        # --------------------------------------------------------
        side_cache = tq_state.ensure_side_key_cache(
            num_layers=len(self._turboquant_parent_model.model.layers)
        )

        # Append only the newly produced token(s) K into TurboQuant side cache.
        side_cache.append(
            layer_idx=layer_idx,
            key_states=key_states,
            value_states=None,
        )

        # Maintain dense K/V cache behavior for Hugging Face generation.
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                }

                key_states, value_states = past_key_value.update(
                    key_states=key_states,
                    value_states=value_states,
                    layer_idx=layer_idx,
                    cache_kwargs=cache_kwargs,
                )

                present_key_value = past_key_value if use_cache else None

            elif isinstance(past_key_value, tuple):
                key_states = torch.cat(
                    [past_key_value[0], key_states],
                    dim=2,
                )
                value_states = torch.cat(
                    [past_key_value[1], value_states],
                    dim=2,
                )

                present_key_value = (
                    (key_states, value_states)
                    if use_cache
                    else None
                )

        else:
            present_key_value = (
                (key_states, value_states)
                if use_cache
                else None
            )

        key_states = repeat_kv(
            key_states,
            self.num_key_value_groups,
        )
        value_states = repeat_kv(
            value_states,
            self.num_key_value_groups,
        )

        attn_weights = _turboquant_scores_from_key_cache(
            layer_idx=layer_idx,
            query_states=query_states,
            tq_state=tq_state,
        )

    # ------------------------------------------------------------
    # Scale + mask
    # ------------------------------------------------------------
    kv_score_len = attn_weights.shape[-1]

    with _nvtx_range("tq_mask_scale"):
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            _validate_attention_mask_shape(
                attention_mask=attention_mask,
                bsz=bsz,
                q_len=q_len,
                kv_seq_len=kv_score_len,
            )

            attn_weights = attn_weights + attention_mask

    # ------------------------------------------------------------
    # Softmax + dtype cast
    # ------------------------------------------------------------
    with _nvtx_range("tq_softmax_cast"):
        attn_weights = nn.functional.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

    # ------------------------------------------------------------
    # Attention × V
    # ------------------------------------------------------------
    with _nvtx_range("tq_attn_value_matmul"):
        attn_output = torch.matmul(
            attn_weights,
            value_states,
        )

    expected_attn_shape = (
        bsz,
        self.num_heads,
        q_len,
        self.head_dim,
    )

    if attn_output.size() != expected_attn_shape:
        raise ValueError(
            f"`attn_output` should be of size {expected_attn_shape}, "
            f"but is {tuple(attn_output.size())}"
        )

    # ------------------------------------------------------------
    # Output projection
    # ------------------------------------------------------------
    attn_output = _apply_output_projection(
        self,
        attn_output,
    )

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, present_key_value


# ============================================================
# Public patch API
# ============================================================

def patch_llama_model_with_turboquant_scores(
    model,
    qjl_m: int = 256,
    device: str | torch.device = "cuda:0",
    use_cuda_decode_score: bool = True,
    compare_cuda_decode_score: bool = False,
):
    """
    Patch all LLaMA self-attention modules in-place.

    Returns:
      TurboQuantScoreState
    """
    layers = model.model.layers

    if len(layers) == 0:
        raise RuntimeError("Model has no decoder layers.")

    first_attn = layers[0].self_attn
    head_dim = int(first_attn.head_dim)

    tq_state = TurboQuantScoreState(
        head_dim=head_dim,
        qjl_m=qjl_m,
        device=device,
        dtype=torch.float32,
        use_cuda_decode_score=use_cuda_decode_score,
        compare_cuda_decode_score=compare_cuda_decode_score,
    )

    patched = 0

    for layer_idx, decoder_layer in enumerate(layers):
        attn = decoder_layer.self_attn

        # Keep layer_idx stable.
        attn.layer_idx = layer_idx

        # Attach shared state.
        attn._turboquant_score_state = tq_state
        attn._turboquant_parent_model = model

        # Save original forward only once.
        if not hasattr(attn, "_turboquant_original_forward"):
            attn._turboquant_original_forward = attn.forward

        # Monkey-patch forward.
        attn.forward = types.MethodType(
            _turboquant_llama_attention_forward,
            attn,
        )

        patched += 1

    print(
        f"[TurboQuantScorePatch] Patched {patched} "
        f"Llama self-attention modules."
    )

    return tq_state