from __future__ import annotations

import math
import types
from typing import Any

import torch
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from turboquant.runtime_cache import (
    TurboQuantRuntimeCache,
)
from turboquant.polar_runtime_cache import (
    TurboQuantPolarRuntimeCache,
)


def _safe_get_cache_seq_len(
    past_key_value: Any,
    layer_idx: int,
) -> int:
    """
    Get cached sequence length from either:
      - custom TurboQuant runtime cache
      - HuggingFace Cache
      - legacy tuple cache

    Returns 0 if cache is empty / unavailable.
    """
    if past_key_value is None:
        return 0

    if hasattr(past_key_value, "get_seq_length"):
        try:
            return int(past_key_value.get_seq_length(layer_idx))
        except TypeError:
            try:
                return int(past_key_value.get_seq_length())
            except Exception:
                pass
        except Exception:
            pass

    # Legacy tuple/list style:
    # past_key_value[layer_idx] = (K, V)
    try:
        maybe_layer_cache = past_key_value[layer_idx]
        if maybe_layer_cache is not None:
            maybe_k = maybe_layer_cache[0]
            if torch.is_tensor(maybe_k):
                return int(maybe_k.shape[-2])
    except Exception:
        pass

    return 0


def _update_standard_cache(
    *,
    past_key_value: Any,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Default HuggingFace cache update path.
    """
    if past_key_value is None:
        return key_states, value_states

    if hasattr(past_key_value, "update"):
        return past_key_value.update(
            key_states,
            value_states,
            layer_idx,
            cache_kwargs,
        )

    # Legacy tuple cache should not normally reach here in the newer
    # Cache-based generation path, but keep a defensive fallback.
    return key_states, value_states


def _maybe_repeat_kv_for_custom_cache(
    *,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom TurboQuant caches score against query heads, so K/V must match
    query head count. For ordinary LLaMA-7B this is a no-op because
    num_key_value_groups == 1.
    """
    if num_key_value_groups == 1:
        return key_states, value_states

    return (
        repeat_kv(key_states, num_key_value_groups),
        repeat_kv(value_states, num_key_value_groups),
    )


def _attention_from_scores(
    *,
    self_attn: LlamaAttention,
    attn_weights: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    bsz: int,
    q_len: int,
    output_attentions: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Complete the attention path after qK^T scores are available.

    attn_weights:
        [B, H, Q, K]

    value_states:
        [B, H, K, D]
    """
    attn_weights = attn_weights / math.sqrt(self_attn.head_dim)

    if attention_mask is not None:
        # Expected broadcast-compatible shape:
        # [B, 1, Q, K]
        attn_weights = attn_weights + attention_mask

    # Match standard LLaMA path: softmax in fp32 then cast back.
    attn_weights = F.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    ).to(value_states.dtype)

    attn_output = torch.matmul(
        attn_weights,
        value_states,
    )

    expected = (
        bsz,
        self_attn.num_heads,
        q_len,
        self_attn.head_dim,
    )

    if tuple(attn_output.shape) != expected:
        raise ValueError(
            f"`attn_output` should be {expected}, got {tuple(attn_output.shape)}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(
        bsz,
        q_len,
        self_attn.hidden_size,
    )

    if getattr(self_attn.config, "pretraining_tp", 1) > 1:
        raise NotImplementedError(
            "TurboQuant score patch currently expects pretraining_tp == 1."
        )

    attn_output = self_attn.o_proj(attn_output)

    if not output_attentions:
        attn_weights_out = None
    else:
        attn_weights_out = attn_weights

    return attn_output, attn_weights_out


def _turboquant_polar_forward(
    self: LlamaAttention,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None,
    past_key_value: TurboQuantPolarRuntimeCache,
    output_attentions: bool,
    use_cache: bool,
    **kwargs: Any,
):
    """
    Forward path for:
      TurboQuantPolarRuntimeCache
    """
    del kwargs

    bsz, q_len, _ = hidden_states.size()

    if getattr(self.config, "pretraining_tp", 1) > 1:
        raise NotImplementedError(
            "TurboQuant polar runtime patch currently expects pretraining_tp == 1."
        )

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz,
        q_len,
        self.num_heads,
        self.head_dim,
    ).transpose(1, 2)

    key_states = key_states.view(
        bsz,
        q_len,
        self.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)

    value_states = value_states.view(
        bsz,
        q_len,
        self.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)

    layer_idx = int(self.layer_idx)

    past_len = _safe_get_cache_seq_len(
        past_key_value,
        layer_idx,
    )

    kv_seq_len = int(key_states.shape[-2]) + int(past_len)

    cos, sin = self.rotary_emb(
        value_states,
        seq_len=kv_seq_len,
    )

    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        position_ids,
    )

    key_states, value_states = _maybe_repeat_kv_for_custom_cache(
        key_states=key_states,
        value_states=value_states,
        num_key_value_groups=self.num_key_value_groups,
    )

    if use_cache:
        past_key_value.update(
            key_states,
            value_states,
            layer_idx,
            cache_kwargs={
                "sin": sin,
                "cos": cos,
            },
        )

        attn_weights = past_key_value.score(
            layer_idx=layer_idx,
            query_states=query_states,
        )

        value_states_full = past_key_value.get_value_states(
            layer_idx=layer_idx,
        )
    else:
        # This branch is mostly defensive; the custom compressed cache is
        # designed for use_cache=True generation.
        temp_cache = TurboQuantPolarRuntimeCache(
            num_layers=1,
            codebooks=past_key_value.codebooks,
            sketch=past_key_value.sketch,
            num_levels=past_key_value.num_levels,
        )
        temp_cache.update(
            key_states,
            value_states,
            layer_idx=0,
            cache_kwargs=None,
        )
        attn_weights = temp_cache.score(
            layer_idx=0,
            query_states=query_states,
        )
        value_states_full = temp_cache.get_value_states(
            layer_idx=0,
        )

    attn_output, attn_weights_out = _attention_from_scores(
        self_attn=self,
        attn_weights=attn_weights,
        value_states=value_states_full,
        attention_mask=attention_mask,
        bsz=bsz,
        q_len=q_len,
        output_attentions=output_attentions,
    )

    return attn_output, attn_weights_out, past_key_value


def _turboquant_legacy_runtime_forward(
    self: LlamaAttention,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None,
    past_key_value: TurboQuantRuntimeCache,
    output_attentions: bool,
    use_cache: bool,
    **kwargs: Any,
):
    """
    Forward path for the existing pre-Polar TurboQuantRuntimeCache.

    This assumes the existing runtime cache already provides:
      - update(...)
      - score(...)
      - get_value_states(...)
    """
    del kwargs

    bsz, q_len, _ = hidden_states.size()

    if getattr(self.config, "pretraining_tp", 1) > 1:
        raise NotImplementedError(
            "TurboQuant runtime patch currently expects pretraining_tp == 1."
        )

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz,
        q_len,
        self.num_heads,
        self.head_dim,
    ).transpose(1, 2)

    key_states = key_states.view(
        bsz,
        q_len,
        self.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)

    value_states = value_states.view(
        bsz,
        q_len,
        self.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)

    layer_idx = int(self.layer_idx)

    past_len = _safe_get_cache_seq_len(
        past_key_value,
        layer_idx,
    )

    kv_seq_len = int(key_states.shape[-2]) + int(past_len)

    cos, sin = self.rotary_emb(
        value_states,
        seq_len=kv_seq_len,
    )

    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        position_ids,
    )

    key_states, value_states = _maybe_repeat_kv_for_custom_cache(
        key_states=key_states,
        value_states=value_states,
        num_key_value_groups=self.num_key_value_groups,
    )

    if use_cache:
        past_key_value.update(
            key_states,
            value_states,
            layer_idx,
            cache_kwargs={
                "sin": sin,
                "cos": cos,
            },
        )

        attn_weights = past_key_value.score(
            layer_idx=layer_idx,
            query_states=query_states,
        )

        value_states_full = past_key_value.get_value_states(
            layer_idx=layer_idx,
        )
    else:
        raise RuntimeError(
            "TurboQuantRuntimeCache path expects use_cache=True."
        )

    attn_output, attn_weights_out = _attention_from_scores(
        self_attn=self,
        attn_weights=attn_weights,
        value_states=value_states_full,
        attention_mask=attention_mask,
        bsz=bsz,
        q_len=q_len,
        output_attentions=output_attentions,
    )

    return attn_output, attn_weights_out, past_key_value


def _standard_llama_forward_fallback(
    self: LlamaAttention,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None,
    past_key_value: Any,
    output_attentions: bool,
    use_cache: bool,
    **kwargs: Any,
):
    """
    Standard LLaMA attention fallback for non-TurboQuant cache.
    """
    del kwargs

    bsz, q_len, _ = hidden_states.size()

    if getattr(self.config, "pretraining_tp", 1) > 1:
        raise NotImplementedError(
            "This replacement llama_score_patch.py expects pretraining_tp == 1."
        )

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz,
        q_len,
        self.num_heads,
        self.head_dim,
    ).transpose(1, 2)

    key_states = key_states.view(
        bsz,
        q_len,
        self.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)

    value_states = value_states.view(
        bsz,
        q_len,
        self.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)

    layer_idx = int(self.layer_idx)

    past_len = _safe_get_cache_seq_len(
        past_key_value,
        layer_idx,
    )

    kv_seq_len = int(key_states.shape[-2]) + int(past_len)

    cos, sin = self.rotary_emb(
        value_states,
        seq_len=kv_seq_len,
    )

    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        position_ids,
    )

    if past_key_value is not None and use_cache:
        key_states, value_states = _update_standard_cache(
            past_key_value=past_key_value,
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            cache_kwargs={
                "sin": sin,
                "cos": cos,
            },
        )

    key_states = repeat_kv(
        key_states,
        self.num_key_value_groups,
    )

    value_states = repeat_kv(
        value_states,
        self.num_key_value_groups,
    )

    attn_weights = torch.matmul(
        query_states,
        key_states.transpose(2, 3),
    )

    attn_output, attn_weights_out = _attention_from_scores(
        self_attn=self,
        attn_weights=attn_weights,
        value_states=value_states,
        attention_mask=attention_mask,
        bsz=bsz,
        q_len=q_len,
        output_attentions=output_attentions,
    )

    return attn_output, attn_weights_out, past_key_value


def turboquant_llama_attention_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Any | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs: Any,
):
    """
    Patched LlamaAttention.forward dispatcher.

    Priority:
      1. TurboQuantPolarRuntimeCache
      2. TurboQuantRuntimeCache
      3. standard LLaMA fallback
    """
    if isinstance(
        past_key_value,
        TurboQuantPolarRuntimeCache,
    ):
        return _turboquant_polar_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

    if isinstance(
        past_key_value,
        TurboQuantRuntimeCache,
    ):
        return _turboquant_legacy_runtime_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

    return _standard_llama_forward_fallback(
        self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )


def patch_llama_self_attention_for_turboquant_scores(
    model: torch.nn.Module,
) -> int:
    """
    Patch all LlamaAttention modules in-place.

    Returns:
        number of patched attention modules.
    """
    patched = 0

    for module in model.modules():
        if isinstance(module, LlamaAttention):
            module.forward = types.MethodType(
                turboquant_llama_attention_forward,
                module,
            )
            patched += 1

    print(
        f"[TurboQuantScorePatch] Patched {patched} Llama self-attention modules."
    )

    return patched


# Backward-compatible alias in case older tests/scripts imported
# a shorter patch helper name.
patch_llama_attention_for_turboquant_scores = (
    patch_llama_self_attention_for_turboquant_scores
)

def patch_llama_model_with_turboquant_scores(
    model: torch.nn.Module,
    *args,
    **kwargs,
) -> dict[str, object]:
    """
    Backward-compatible patch entrypoint.

    Older tests/scripts pass arguments such as:
      qjl_m=...
      rotation_seed=...
      sketch_seed=...

    The current patcher only installs the attention forward dispatcher.
    Polar runtime cache construction is handled separately by the caller.
    """
    del args
    del kwargs

    patched = patch_llama_self_attention_for_turboquant_scores(
        model
    )

    return {
        "patched_modules": int(patched),
    }