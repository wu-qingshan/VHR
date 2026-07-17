"""Qwen2.5-VL-specific VHR attention adapter.

The historical :mod:`vhr` module patches LLaMA attention globally. Qwen2.5-VL
uses GQA and multimodal RoPE, so this module deliberately has a separate,
instance-local adapter. With phase ``off`` or ``vanilla`` it delegates to the
original Hugging Face forward method. Only explicitly selected text decoder
layers are modified.
"""

from types import MethodType
from typing import Iterable, Optional, Tuple

import math
import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


VALID_PHASES = {"off", "text", "clean", "decode", "vanilla"}


def _clear_attention_state(attention):
    attention._qvhr_phase = "off"
    attention._qvhr_text_head_output = None
    attention._qvhr_selected_heads = None
    attention._qvhr_last_head_vary = None
    attention._qvhr_prefill_counts = {"text": 0, "clean": 0}


def qwen_vhr_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """Eager Qwen attention with VHR applied after GQA expansion.

    ``text`` stores the last-token per-query-head output from a text-only
    prefill. ``clean`` selects visual heads from the clean/text difference and
    scales them. ``decode`` reuses the current sample's selection. ``vanilla``
    is used by the base branch of self-contrast and never mutates VHR state.
    """

    phase = getattr(self, "_qvhr_phase", "off")
    if phase not in VALID_PHASES:
        raise RuntimeError(f"unknown Qwen VHR phase: {phase!r}")
    if phase in {"off", "vanilla"}:
        return self._qvhr_original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    if position_embeddings is None:
        raise RuntimeError("Qwen VHR requires model-provided multimodal RoPE embeddings")
    if hidden_states.shape[0] != 1:
        raise NotImplementedError("Qwen VHR engineering stage supports batch size 1 only")

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
        )

    # Qwen2.5-VL-7B has 28 query heads and four KV heads. VHR selection is
    # defined over query heads, so it must occur after this repeat.
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    if query_states.dtype == torch.float16:
        attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights,
        p=self.attention_dropout,
        training=self.training,
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"unexpected Qwen attention output {tuple(attn_output.size())}; "
            f"expected {(bsz, self.num_heads, q_len, self.head_dim)}"
        )
    if not torch.isfinite(attn_output).all():
        raise FloatingPointError(f"non-finite Qwen attention output in layer {self.layer_idx}")

    if phase == "text":
        if q_len <= 1:
            raise RuntimeError("Qwen VHR text calibration must be a prefill")
        self._qvhr_text_head_output = attn_output[0, :, -1, :].detach().clone()
        self._qvhr_prefill_counts["text"] += 1
    elif phase == "clean":
        if q_len <= 1:
            raise RuntimeError("Qwen VHR clean calibration must be a prefill")
        text_output = self._qvhr_text_head_output
        if text_output is None:
            raise RuntimeError(f"Qwen VHR clean prefill reached layer {self.layer_idx} before text calibration")
        clean_output = attn_output[0, :, -1, :]
        head_vary = (clean_output.float() - text_output.float()).pow(2).sum(dim=-1)
        if getattr(self, "_qvhr_filter", True) and head_vary.std() > 1:
            text_activity = text_output.float().pow(2).sum(dim=-1)
            outliers = (
                (head_vary > head_vary.mean() + head_vary.std())
                & (text_activity > text_activity.mean() + text_activity.std())
            )
            head_vary = head_vary.masked_fill(outliers, 0)
        selected = (head_vary > head_vary.median()).nonzero().flatten()
        if selected.numel() == 0:
            selected = head_vary.argmax().view(1)
        self._qvhr_selected_heads = selected
        self._qvhr_last_head_vary = head_vary.detach().cpu()
        self._qvhr_text_head_output = None
        self._qvhr_prefill_counts["clean"] += 1
        attn_output[:, selected, :, :] *= self._qvhr_aug_ratio
    elif phase == "decode":
        if q_len != 1:
            raise RuntimeError(f"Qwen VHR decode expected q_len=1, got {q_len}")
        selected = self._qvhr_selected_heads
        if selected is None:
            raise RuntimeError(f"Qwen VHR decode reached layer {self.layer_idx} without selected heads")
        attn_output[:, selected, :, :] *= self._qvhr_aug_ratio

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def get_qwen_text_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected Qwen2.5-VL model.model.layers")
    return layers


def install_qwen_vhr(
    model,
    target_layers: Iterable[int],
    aug_ratio: float = 2.0,
    use_filter: bool = True,
):
    """Install the instance-local eager adapter on the requested layers."""

    layers = get_qwen_text_layers(model)
    target_layers = tuple(sorted(set(int(index) for index in target_layers)))
    if not target_layers:
        raise ValueError("target_layers must not be empty")
    if target_layers[0] < 0 or target_layers[-1] >= len(layers):
        raise IndexError(f"target layers {target_layers} outside 0..{len(layers) - 1}")
    if aug_ratio <= 0:
        raise ValueError(f"aug_ratio must be positive, got {aug_ratio}")

    for index in target_layers:
        attention = layers[index].self_attn
        if type(attention) is not Qwen2_5_VLAttention:
            raise TypeError(
                f"layer {index} uses {type(attention).__name__}; "
                "the preregistered Qwen VHR adapter requires eager attention"
            )
        if hasattr(attention, "_qvhr_original_forward"):
            raise RuntimeError(f"Qwen VHR already installed on layer {index}")
        attention._qvhr_original_forward = attention.forward
        attention._qvhr_aug_ratio = float(aug_ratio)
        attention._qvhr_filter = bool(use_filter)
        _clear_attention_state(attention)
        attention.forward = MethodType(qwen_vhr_forward, attention)
    model._qvhr_target_layers = target_layers
    return target_layers


def set_qwen_vhr_phase(model, phase: str):
    if phase not in VALID_PHASES:
        raise ValueError(f"unknown Qwen VHR phase: {phase!r}")
    layers = get_qwen_text_layers(model)
    for index in getattr(model, "_qvhr_target_layers", ()):
        layers[index].self_attn._qvhr_phase = phase


def reset_qwen_vhr_state(model):
    layers = get_qwen_text_layers(model)
    for index in getattr(model, "_qvhr_target_layers", ()):
        _clear_attention_state(layers[index].self_attn)
    if hasattr(model, "rope_deltas"):
        model.rope_deltas = None


def qwen_vhr_state_report(model):
    layers = get_qwen_text_layers(model)
    report = []
    for index in getattr(model, "_qvhr_target_layers", ()):
        attention = layers[index].self_attn
        selected = attention._qvhr_selected_heads
        report.append(
            {
                "layer": index,
                "phase": attention._qvhr_phase,
                "selected_heads": None if selected is None else selected.detach().cpu().tolist(),
                "prefill_counts": dict(attention._qvhr_prefill_counts),
                "finite_head_vary": (
                    None
                    if attention._qvhr_last_head_vary is None
                    else bool(torch.isfinite(attention._qvhr_last_head_vary).all())
                ),
            }
        )
    return report


def uninstall_qwen_vhr(model):
    layers = get_qwen_text_layers(model)
    for index in getattr(model, "_qvhr_target_layers", ()):
        attention = layers[index].self_attn
        original = getattr(attention, "_qvhr_original_forward", None)
        if original is None:
            continue
        attention.forward = original
        for name in (
            "_qvhr_original_forward",
            "_qvhr_aug_ratio",
            "_qvhr_filter",
            "_qvhr_phase",
            "_qvhr_text_head_output",
            "_qvhr_selected_heads",
            "_qvhr_last_head_vary",
            "_qvhr_prefill_counts",
        ):
            if hasattr(attention, name):
                delattr(attention, name)
    if hasattr(model, "_qvhr_target_layers"):
        delattr(model, "_qvhr_target_layers")
    if hasattr(model, "rope_deltas"):
        model.rope_deltas = None
