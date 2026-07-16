from functools import partial
from typing import Optional, Tuple

import math
import torch
import transformers
from transformers.models.llama.modeling_llama import *
from transformers.cache_utils import Cache

from utils import get_layers


def vhr_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    aug_ratio: float = 1,
    filter: bool = False,
    score_mode: str = "magnitude",
    head_fraction: float = 0.5,
    positive_only: bool = False,
    use_avhr: bool = False,
    avhr_tau: float = 1.0,
    avhr_rho_max: float = 2.0,
    avhr_gamma_l: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    VHR / A-VHR forward.

    - 默认 (use_avhr=False): 论文原版 VHR，固定 aug_ratio + 中位数硬阈值。
    - 启用 (use_avhr=True): A-VHR (Adaptive-ratio VHR)，按下式给每个 head 一个连续 ratio：
            r_h = 1 + (rho_max - 1) * gamma_l * sigmoid((v_h - mu) / (tau * s))
        其中 v_h 是 head_vary，mu / s 为层内均值和标准差，gamma_l 是层深衰减系数（在
        VHR 作用范围内，越靠近输出层越小，使顶层更保守）。
    """
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    now_attn_output = attn_output[0, :, -1, :]

    # ---------- VHR / A-VHR head re-weighting ----------
    # CC-VHR uses explicit phases because it has three prefill branches.  The
    # original two-view methods keep the legacy first-prefill/second-prefill
    # behavior below for exact backward compatibility.
    if (
        score_mode != "consensus"
        and attn_output.shape[-2] > 1
        and getattr(self, "vhr_phase", None) == "reuse"
    ):
        # Symmetric VHR-VCD noisy prefill: reuse the heads selected by the
        # clean prefill without treating this third branch as a new contrast.
        if getattr(self, "aug_ratio_per_head", None) is not None:
            attn_output = attn_output * self.aug_ratio_per_head
        elif getattr(self, "aug_heads", None) is not None:
            attn_output[:, self.aug_heads, :, :] = aug_ratio * attn_output[:, self.aug_heads, :, :]
        else:
            raise RuntimeError("VHR-VCD noisy prefill ran before clean head selection")

    elif attn_output.shape[-2] > 1 and getattr(self, "vhr_phase", None) == "vanilla":
        # Self-contrast baseline prefill: keep the original attention output
        # and do not overwrite any VHR calibration state.
        pass

    elif score_mode == "consensus" and attn_output.shape[-2] > 1:
        phase = getattr(self, "vhr_phase", None)
        if phase == "text":
            self.attn_output_tc = now_attn_output
        elif phase == "noisy":
            self.attn_output_nc = now_attn_output
        elif phase == "clean":
            if getattr(self, "attn_output_tc", None) is None or getattr(self, "attn_output_nc", None) is None:
                raise RuntimeError("CC-VHR clean prefill requires both text and noisy head summaries")

            text_delta = now_attn_output.float() - self.attn_output_tc.float()
            noise_delta = now_attn_output.float() - self.attn_output_nc.float()
            text_strength = text_delta.norm(dim=-1)
            noise_strength = noise_delta.norm(dim=-1)
            alignment = torch.nn.functional.cosine_similarity(text_delta, noise_delta, dim=-1, eps=1e-6)
            harmonic_strength = (
                2.0 * text_strength * noise_strength / (text_strength + noise_strength + 1e-6)
            )
            scores = alignment * harmonic_strength

            if filter and text_strength.std() > 1:
                text_activity = self.attn_output_tc.float().pow(2).sum(-1)
                outliers = (
                    (text_strength > text_strength.mean() + text_strength.std())
                    & (text_activity > text_activity.mean() + text_activity.std())
                )
                scores = scores.masked_fill(outliers, -float("inf"))

            if not 0 < head_fraction <= 1:
                raise ValueError(f"head_fraction must be in (0, 1], got {head_fraction}")
            max_heads = max(1, min(scores.numel(), math.ceil(scores.numel() * head_fraction)))
            if positive_only:
                eligible = (scores > 0).nonzero().flatten()
                if eligible.numel() == 0:
                    self.aug_heads = scores.argmax().view(1)
                else:
                    k = min(max_heads, eligible.numel())
                    self.aug_heads = eligible[torch.topk(scores[eligible], k=k).indices]
            else:
                self.aug_heads = torch.topk(scores, k=max_heads).indices
            self.aug_ratio_per_head = None
            self.attn_output_tc = None
            self.attn_output_nc = None
            attn_output[:, self.aug_heads, :, :] = aug_ratio * attn_output[:, self.aug_heads, :, :]
        else:
            raise RuntimeError(f"Unknown CC-VHR prefill phase: {phase!r}")

    elif attn_output.shape[-2] > 1 and (not hasattr(self, "attn_output_tc") or self.attn_output_tc is None):
        # 第一次 prefill：缓存 text-contrast 路径的最后一个 token 表示
        self.attn_output_tc = now_attn_output

    elif attn_output.shape[-2] > 1:
        # 第二次 prefill：基于两次 prefill 计算 head_vary，决定本步以及后续 decoding 的加权
        text_attn_output = self.attn_output_tc
        head_delta = now_attn_output - text_attn_output
        head_vary = (head_delta ** 2).sum(-1)
        if head_vary.std() > 1 and filter:
            aotc = (self.attn_output_tc ** 2).sum(-1)
            head_vary[(head_vary > (head_vary.mean() + head_vary.std())) & (aotc > (aotc.mean() + aotc.std()))] = 0
        self.attn_output_tc = None

        if use_avhr:
            mu = head_vary.mean()
            s = head_vary.std() + 1e-6
            gate = torch.sigmoid((head_vary - mu) / (avhr_tau * s))           # [num_heads]
            ratio = 1.0 + (avhr_rho_max - 1.0) * avhr_gamma_l * gate          # [num_heads]
            ratio = ratio.to(attn_output.dtype).view(1, -1, 1, 1)             # [1, H, 1, 1]
            self.aug_ratio_per_head = ratio
            self.aug_heads = None  # 标记走 A-VHR 分支
            attn_output = attn_output * ratio
        else:
            if score_mode in {"directional", "directional_linear", "delayed_directional_linear", "calibrated"}:
                num_heads = head_delta.shape[0]
                o_proj_blocks = self.o_proj.weight.float().view(self.o_proj.out_features, num_heads, self.head_dim)
                projected_delta = torch.einsum("hd,ohd->ho", head_delta.float(), o_proj_blocks)
                visual_residual = projected_delta.sum(dim=0)
                alignment = torch.nn.functional.cosine_similarity(
                    projected_delta,
                    visual_residual.unsqueeze(0),
                    dim=-1,
                    eps=1e-6,
                )
                head_strength = head_delta.float().norm(dim=-1)
                if score_mode in {"directional_linear", "delayed_directional_linear"}:
                    scores = alignment * head_strength
                else:
                    scores = alignment * torch.log1p(head_strength)
                scores = scores.masked_fill(head_vary == 0, -float("inf"))
            else:
                scores = head_vary.float()

            if score_mode == "delayed_directional_linear":
                baseline_heads = (head_vary > head_vary.median()).nonzero().flatten()
                max_heads = max(1, min(scores.numel(), math.ceil(scores.numel() * head_fraction)))
                eligible = (scores > 0).nonzero().flatten() if positive_only else torch.arange(scores.numel(), device=scores.device)
                if eligible.numel() == 0:
                    directional_heads = scores.argmax().view(1)
                else:
                    k = min(max_heads, eligible.numel())
                    directional_heads = eligible[torch.topk(scores[eligible], k=k).indices]
                self.aug_heads = baseline_heads
                self.vhr_baseline_heads = baseline_heads
                self.vhr_directional_heads = directional_heads
                self.vhr_directional_active = False
            elif score_mode == "calibrated":
                baseline_heads = head_vary > head_vary.median()
                eligible = (baseline_heads & (alignment > 0)).nonzero().flatten()
                self.aug_heads = eligible if eligible.numel() > 0 else baseline_heads.nonzero().flatten()
            elif score_mode == "magnitude" and head_fraction == 0.5 and not positive_only:
                self.aug_heads = (head_vary > head_vary.median()).nonzero().flatten()
            else:
                if not 0 < head_fraction <= 1:
                    raise ValueError(f"head_fraction must be in (0, 1], got {head_fraction}")
                max_heads = max(1, min(scores.numel(), math.ceil(scores.numel() * head_fraction)))
                if positive_only:
                    eligible = (scores > 0).nonzero().flatten()
                    if eligible.numel() == 0:
                        self.aug_heads = scores.argmax().view(1)
                    else:
                        k = min(max_heads, eligible.numel())
                        self.aug_heads = eligible[torch.topk(scores[eligible], k=k).indices]
                else:
                    self.aug_heads = torch.topk(scores, k=max_heads).indices
            self.aug_ratio_per_head = None
            attn_output[:, self.aug_heads, :, :] = aug_ratio * attn_output[:, self.aug_heads, :, :]

    elif attn_output.shape[-2] == 1 and getattr(self, "vhr_phase", None) != "vanilla":
        # decoding 步：复用 prefill 时计算好的加权
        if getattr(self, "aug_ratio_per_head", None) is not None:
            attn_output = attn_output * self.aug_ratio_per_head
        elif getattr(self, "aug_heads", None) is not None:
            active_heads = self.aug_heads
            if getattr(self, "vhr_directional_active", False) and getattr(self, "vhr_directional_heads", None) is not None:
                active_heads = self.vhr_directional_heads
            attn_output[:, active_heads, :, :] = aug_ratio * attn_output[:, active_heads, :, :]
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def _compute_gamma_l(layer_idx, target_layers, alpha):
    """在 VHR 作用范围内计算层深衰减系数：
       gamma_l = exp(-alpha * (l - l_start) / max(1, l_end - l_start))
       l_start (作用范围内最小 idx) -> gamma=1
       l_end   (作用范围内最大 idx) -> gamma=exp(-alpha)
    """
    if not target_layers:
        return 1.0
    l_start = min(target_layers)
    l_end = max(target_layers)
    span = max(1, l_end - l_start)
    return math.exp(-alpha * (layer_idx - l_start) / span)


def replace_attn(
    model,
    target_layers=None,
    aug_ratio=2,
    filter=False,
    score_mode="magnitude",
    head_fraction=0.5,
    positive_only=False,
    use_avhr=False,
    avhr_tau=1.0,
    avhr_alpha=0.5,
    avhr_rho_max=2.0,
):
    """绑定每一层的 attention forward。

    Args:
        target_layers: 应用 VHR 的层 idx 列表（与 main.py 计算一致）。
        aug_ratio:    原版 VHR 的固定加权倍率。
        filter:       原版 VHR 的离群头过滤。
        use_avhr:     是否启用 A-VHR（连续头加权 + 层深衰减）。
        avhr_tau:     sigmoid 温度，越小越接近硬门控。
        avhr_alpha:   层深衰减强度，0 表示不衰减；越大顶层越保守。
        avhr_rho_max: 单 head 加权上限 rho_max。
    """
    target_layers = target_layers or []

    forward = transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward

    for idx, layer in enumerate(get_layers(model)):
        if idx in target_layers:
            gamma_l = _compute_gamma_l(idx, target_layers, avhr_alpha) if use_avhr else 1.0
            layer.self_attn.forward = partial(
                vhr_forward,
                layer.self_attn,
                aug_ratio=aug_ratio,
                filter=filter,
                score_mode=score_mode,
                head_fraction=head_fraction,
                positive_only=positive_only,
                use_avhr=use_avhr,
                avhr_tau=avhr_tau,
                avhr_rho_max=avhr_rho_max,
                avhr_gamma_l=gamma_l,
            )
            # 清空可能残留的状态
            layer.self_attn.attn_output_tc = None
            layer.self_attn.attn_output_nc = None
            layer.self_attn.aug_heads = None
            layer.self_attn.aug_ratio_per_head = None
            layer.self_attn.vhr_baseline_heads = None
            layer.self_attn.vhr_directional_heads = None
            layer.self_attn.vhr_directional_active = False
            layer.self_attn.vhr_phase = None
        else:
            layer.self_attn.forward = partial(forward, layer.self_attn)
            layer.self_attn.attn_output_tc = None
            layer.self_attn.attn_output_nc = None
            layer.self_attn.aug_heads = None
            layer.self_attn.aug_ratio_per_head = None
            layer.self_attn.vhr_baseline_heads = None
            layer.self_attn.vhr_directional_heads = None
            layer.self_attn.vhr_directional_active = False
            layer.self_attn.vhr_phase = None


def set_directional_active(model, active):
    for layer in get_layers(model):
        attention = layer.self_attn
        if getattr(attention, "vhr_directional_heads", None) is not None:
            attention.vhr_directional_active = active


def set_vhr_phase(model, phase):
    """Set the explicit prefill/decode branch used by multi-view VHR modes."""
    for layer in get_layers(model):
        layer.self_attn.vhr_phase = phase
