"""Standalone greedy VHR / SC-VHR decoding for Qwen2.5-VL.

This module intentionally avoids the historical Transformers 4.45 generation
monkeypatch. It owns the two branch caches explicitly and keeps Qwen's mRoPE
delta with each branch.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import math
import time
import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from qwen_vhr import qwen_vhr_state_report, reset_qwen_vhr_state, set_qwen_vhr_phase


SUPPORTED_MODES = {"vanilla", "vhr", "scvhr", "srrscvhr"}


def is_binary_question(question):
    normalized = question.strip().lower()
    prefixes = (
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "has ",
        "have ",
        "will ",
        "would ",
    )
    return normalized.startswith(prefixes)


def qwen_rr_mode(question):
    return "vhr" if is_binary_question(question) else "scvhr"


@dataclass
class QwenDecodeResult:
    mode: str
    token_ids: list
    text: str
    first_token_confidence: float
    elapsed_seconds: float
    peak_allocated_gib: float
    terminated: bool
    forward_counts: Dict[str, int]
    vhr_state: list
    routing_stats: Dict[str, object]

    def to_dict(self):
        return asdict(self)


def qwen_target_layers(model):
    """Frozen LLaVA-style policy: layer 1 plus the last 14 of 28 layers."""

    layer_count = len(model.model.layers)
    last = list(range(max(0, layer_count - 14), layer_count))
    return sorted(set([1] + last))


def remove_single_vision_span(input_ids, attention_mask, config):
    """Remove one complete ``vision_start ... vision_end`` span."""

    if input_ids.shape[0] != 1:
        raise NotImplementedError("Qwen VHR engineering stage supports batch size 1 only")
    starts = (input_ids[0] == config.vision_start_token_id).nonzero().flatten()
    ends = (input_ids[0] == config.vision_end_token_id).nonzero().flatten()
    if starts.numel() != 1 or ends.numel() != 1:
        raise ValueError(f"expected one vision span, got starts={starts.tolist()} ends={ends.tolist()}")
    start = int(starts.item())
    end = int(ends.item()) + 1
    if end <= start:
        raise ValueError(f"invalid vision span [{start}, {end})")
    text_ids = torch.cat([input_ids[:, :start], input_ids[:, end:]], dim=1)
    text_mask = torch.cat([attention_mask[:, :start], attention_mask[:, end:]], dim=1)
    if (text_ids == config.image_token_id).any():
        raise AssertionError("image token remained after removing the vision span")
    return text_ids, text_mask, (start, end)


def _clone_rope_deltas(model):
    rope = getattr(model, "rope_deltas", None)
    return None if rope is None else rope.detach().clone()


def _set_rope_deltas(model, rope_deltas):
    model.rope_deltas = None if rope_deltas is None else rope_deltas.detach().clone()


def _model_device(model):
    return next(model.parameters()).device


def _full_prefill_kwargs(inputs):
    allowed = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    }
    return {key: value for key, value in inputs.items() if key in allowed}


@torch.inference_mode()
def _text_calibration(model, inputs):
    text_ids, text_mask, span = remove_single_vision_span(
        inputs["input_ids"],
        inputs["attention_mask"],
        model.config,
    )
    model.rope_deltas = None
    set_qwen_vhr_phase(model, "text")
    outputs = model(
        input_ids=text_ids,
        attention_mask=text_mask,
        use_cache=False,
        return_dict=True,
    )
    if not torch.isfinite(outputs.logits[:, -1, :]).all():
        raise FloatingPointError("non-finite logits in Qwen text calibration")
    return span


@torch.inference_mode()
def _prefill_branch(model, inputs, phase):
    model.rope_deltas = None
    set_qwen_vhr_phase(model, phase)
    outputs = model(
        **_full_prefill_kwargs(inputs),
        use_cache=True,
        return_dict=True,
    )
    logits = outputs.logits[:, -1, :].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f"non-finite logits in Qwen {phase} prefill")
    return logits, outputs.past_key_values, _clone_rope_deltas(model)


@torch.inference_mode()
def _decode_branch(model, token, attention_mask, cache, rope_deltas, phase):
    _set_rope_deltas(model, rope_deltas)
    set_qwen_vhr_phase(model, phase)
    cache_position = torch.arange(
        cache.get_seq_length(),
        cache.get_seq_length() + token.shape[1],
        device=token.device,
    )
    outputs = model(
        input_ids=token,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        return_dict=True,
    )
    logits = outputs.logits[:, -1, :].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f"non-finite logits in Qwen {phase} decode")
    return logits, outputs.past_key_values


def _eos_ids(model):
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = model.config.eos_token_id
    if isinstance(eos, int):
        return {eos}
    return set(eos)


def _repetition_processor(model):
    penalty = float(getattr(model.generation_config, "repetition_penalty", 1.0))
    return None if penalty == 1.0 else RepetitionPenaltyLogitsProcessor(penalty)


def cycle_if_appended(processor, generated_ids, candidate_id, max_period=16, repeats=3):
    """Return the exact whitespace cycle completed by one candidate, if any."""

    text = processor.decode(
        list(generated_ids) + [int(candidate_id)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    tokens = text.split()
    max_candidate_period = min(max_period, len(tokens) // repeats)
    for period in range(1, max_candidate_period + 1):
        block = tokens[-period:]
        if all(
            tokens[-(repeat_index + 1) * period : -repeat_index * period] == block
            for repeat_index in range(1, repeats)
        ):
            return {"period": period, "block": block, "text_token_count": len(tokens)}
    return None


def highest_safe_token(scores, processor, generated_ids, max_period=16, repeats=3):
    """Choose the highest-scoring token that does not complete an exact cycle."""

    safe_scores = scores.clone()
    masked = 0
    while masked < safe_scores.shape[-1]:
        token = safe_scores.argmax(dim=-1, keepdim=True)
        cycle = cycle_if_appended(
            processor,
            generated_ids,
            int(token.item()),
            max_period=max_period,
            repeats=repeats,
        )
        if cycle is None:
            return token, masked
        safe_scores.scatter_(-1, token, -float("inf"))
        masked += 1
    raise RuntimeError("all vocabulary tokens were rejected by the exact-cycle safety mask")


@torch.inference_mode()
def generate_qwen(
    model,
    processor,
    inputs,
    mode: str,
    max_new_tokens: int,
    alpha: float = 0.5,
    beta: float = 0.1,
):
    """Generate one sample with explicit Qwen branch/cache ownership."""

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported mode {mode!r}; expected one of {sorted(SUPPORTED_MODES)}")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")

    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    reset_qwen_vhr_state(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    forward_counts = {"text": 0, "vhr": 0, "base": 0}
    generated = []
    first_confidence = None
    terminated = False

    try:
        if mode in {"vhr", "scvhr", "srrscvhr"}:
            _text_calibration(model, inputs)
            forward_counts["text"] += 1
            logits_vhr, cache_vhr, rope_vhr = _prefill_branch(model, inputs, "clean")
            forward_counts["vhr"] += 1
        else:
            logits_vhr, cache_vhr, rope_vhr = _prefill_branch(model, inputs, "vanilla")
            forward_counts["base"] += 1

        cache_base = None
        rope_base = None
        logits_base = None
        if mode in {"scvhr", "srrscvhr"}:
            logits_base, cache_base, rope_base = _prefill_branch(model, inputs, "vanilla")
            forward_counts["base"] += 1

        attention_mask = inputs["attention_mask"].clone()
        input_history = inputs["input_ids"].clone()
        eos_ids = _eos_ids(model)
        repetition_processor = _repetition_processor(model)
        rollback_active = False
        rollback_step = None
        rollback_cycle = None
        safe_masked_tokens = 0

        for step in range(max_new_tokens):
            if mode in {"scvhr", "srrscvhr"} and not rollback_active:
                cutoff = math.log(beta) + logits_vhr.max(dim=-1, keepdim=True).values
                scores = (1 + alpha) * logits_vhr - alpha * logits_base
                scores = scores.masked_fill(logits_vhr < cutoff, -float("inf"))
            elif mode == "srrscvhr" and rollback_active:
                scores = logits_base
            else:
                scores = logits_vhr
            if repetition_processor is not None:
                scores = repetition_processor(input_history, scores)

            token = scores.argmax(dim=-1, keepdim=True)
            if mode == "srrscvhr":
                if not rollback_active:
                    risk = cycle_if_appended(processor, generated, int(token.item()))
                    if risk is not None:
                        rollback_active = True
                        rollback_step = step
                        rollback_cycle = risk
                        base_scores = logits_base
                        if repetition_processor is not None:
                            base_scores = repetition_processor(input_history, base_scores)
                        token, masked = highest_safe_token(base_scores, processor, generated)
                        scores = base_scores
                        safe_masked_tokens += masked
                else:
                    token, masked = highest_safe_token(scores, processor, generated)
                    safe_masked_tokens += masked
            if first_confidence is None:
                first_confidence = float(scores.softmax(dim=-1).gather(-1, token).item())
            token_id = int(token.item())
            generated.append(token_id)
            input_history = torch.cat([input_history, token], dim=1)
            if token_id in eos_ids:
                terminated = True
                break
            if step + 1 == max_new_tokens:
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(token, dtype=attention_mask.dtype)],
                dim=1,
            )
            update_primary = mode in {"vanilla", "vhr", "scvhr"} or (
                mode == "srrscvhr" and not rollback_active
            )
            if update_primary:
                phase = "decode" if mode in {"vhr", "scvhr", "srrscvhr"} else "vanilla"
                logits_vhr, cache_vhr = _decode_branch(
                    model,
                    token,
                    attention_mask,
                    cache_vhr,
                    rope_vhr,
                    phase,
                )
                forward_counts["vhr" if phase == "decode" else "base"] += 1
            if mode in {"scvhr", "srrscvhr"}:
                logits_base, cache_base = _decode_branch(
                    model,
                    token,
                    attention_mask,
                    cache_base,
                    rope_base,
                    "vanilla",
                )
                forward_counts["base"] += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        state = qwen_vhr_state_report(model)
        text = processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        return QwenDecodeResult(
            mode=mode,
            token_ids=generated,
            text=text,
            first_token_confidence=first_confidence or 0.0,
            elapsed_seconds=elapsed,
            peak_allocated_gib=peak,
            terminated=terminated,
            forward_counts=forward_counts,
            vhr_state=state,
            routing_stats={
                "rollback_triggered": rollback_active,
                "rollback_step": rollback_step,
                "rollback_cycle": rollback_cycle,
                "safe_masked_tokens": safe_masked_tokens,
            },
        )
    finally:
        reset_qwen_vhr_state(model)
