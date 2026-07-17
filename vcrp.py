"""Visual-consistent residual projection for VHR mechanism diagnostics."""

from functools import partial

import torch
import transformers

from utils import get_layers


def project_ffn_to_visual_halfspace(ffn_output, visual_residual, eps=1e-6):
    """Return the minimum-L2 FFN correction with non-negative visual alignment."""
    ffn = ffn_output.float()
    visual = visual_residual.float()
    dot = (ffn * visual).sum(dim=-1, keepdim=True)
    visual_norm_sq = visual.pow(2).sum(dim=-1, keepdim=True)
    coefficient = torch.minimum(dot, torch.zeros_like(dot)) / (visual_norm_sq + eps)
    opposing_component = coefficient * visual
    corrected = ffn - opposing_component
    return corrected.to(ffn_output.dtype), {
        "dot": dot.squeeze(-1),
        "conflict": dot.squeeze(-1) < 0,
        "visual_norm": visual.norm(dim=-1),
        "ffn_norm": ffn.norm(dim=-1),
        "correction_norm": opposing_component.norm(dim=-1),
    }


def vcrp_decoder_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    vcrp_intervention=False,
    **kwargs,
):
    """Llama decoder-layer forward with optional diagnostic/projection logic."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    normalized = self.post_attention_layernorm(hidden_states)
    mlp_output = self.mlp(normalized)
    visual_residual = getattr(self.self_attn, "vcrp_visual_residual", None)
    self.vcrp_last_stats = None
    if visual_residual is not None:
        ffn_last = mlp_output[:, -1, :]
        visual_last = visual_residual[:, -1, :]
        corrected, stats = project_ffn_to_visual_halfspace(ffn_last, visual_last)
        stats["hidden_norm"] = residual[:, -1, :].float().norm(dim=-1)
        self.vcrp_last_stats = {key: value.detach() for key, value in stats.items()}
        if vcrp_intervention and hidden_states.shape[1] == 1:
            mlp_output = mlp_output.clone()
            mlp_output[:, -1, :] = corrected
    hidden_states = residual + mlp_output

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def replace_vcrp_decoder_layers(model, target_layers=None, enabled=False, intervention=False):
    """Install or restore isolated VCRP decoder-layer forwards."""
    target_layers = set(target_layers or [])
    original = transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward
    for index, layer in enumerate(get_layers(model)):
        if enabled and index in target_layers:
            layer.forward = partial(
                vcrp_decoder_layer_forward,
                layer,
                vcrp_intervention=intervention,
            )
        else:
            layer.forward = partial(original, layer)
        layer.vcrp_last_stats = None


def collect_vcrp_step_stats(model):
    """Aggregate and clear per-layer VCRP statistics for the current token step."""
    layer_stats = []
    for layer in get_layers(model):
        current = getattr(layer, "vcrp_last_stats", None)
        if current is not None:
            layer_stats.append(current)
            layer.vcrp_last_stats = None
    if not layer_stats:
        raise RuntimeError("VCRP diagnostic captured no target-layer statistics")

    conflict = torch.stack([item["conflict"].float() for item in layer_stats], dim=0)
    visual_norm = torch.stack([item["visual_norm"] for item in layer_stats], dim=0)
    ffn_norm = torch.stack([item["ffn_norm"] for item in layer_stats], dim=0)
    correction_norm = torch.stack([item["correction_norm"] for item in layer_stats], dim=0)
    hidden_norm = torch.stack([item["hidden_norm"] for item in layer_stats], dim=0)
    if any(
        not torch.isfinite(value).all()
        for value in (visual_norm, ffn_norm, correction_norm, hidden_norm)
    ):
        raise RuntimeError("VCRP diagnostic contains non-finite layer statistics")
    return {
        "target_layer_count": len(layer_stats),
        "conflict_layer_fraction": float(conflict.mean().item()),
        "opposing_ffn_fraction": float(
            (correction_norm.sum() / ffn_norm.sum().clamp_min(1e-12)).item()
        ),
        "visual_ffn_norm_ratio": float(
            (visual_norm.sum() / ffn_norm.sum().clamp_min(1e-12)).item()
        ),
        "projection_hidden_fraction": float(
            (correction_norm.sum() / hidden_norm.sum().clamp_min(1e-12)).item()
        ),
        "zero_visual_layer_fraction": float((visual_norm <= 1e-12).float().mean().item()),
    }

