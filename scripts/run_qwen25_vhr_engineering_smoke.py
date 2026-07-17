#!/usr/bin/env python3
"""Run preregistered Qwen VHR/SC engineering equivalence checks."""

import argparse
import json
import os
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_scvhr import generate_qwen, qwen_rr_mode, qwen_target_layers
from qwen_vhr import install_qwen_vhr, set_qwen_vhr_phase, uninstall_qwen_vhr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--output", default="qwen_vhr_smoke_results/engineering_audit.json")
    return parser.parse_args()


def prepare(processor, image_path, prompt="Please describe this image in detail."):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(image_path).resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def to_device(inputs, device):
    return {key: value.to(device) for key, value in inputs.items()}


@torch.inference_mode()
def first_logits(model, inputs):
    model.rope_deltas = None
    outputs = model(**inputs, use_cache=False, return_dict=True)
    return outputs.logits[:, -1, :].float().cpu()


@torch.inference_mode()
def hf_generate(model, inputs, max_new_tokens):
    model.rope_deltas = None
    generated = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    return generated[0, inputs["input_ids"].shape[1] :].cpu().tolist()


def set_ratio(model, ratio):
    for index in model._qvhr_target_layers:
        model.model.layers[index].self_attn._qvhr_aug_ratio = float(ratio)


def state_is_valid(result, target_layers, num_heads):
    if [row["layer"] for row in result.vhr_state] != list(target_layers):
        return False
    for row in result.vhr_state:
        selected = row["selected_heads"]
        if row["prefill_counts"] != {"text": 1, "clean": 1}:
            return False
        if not selected or min(selected) < 0 or max(selected) >= num_heads:
            return False
        if row["finite_head_vary"] is not True:
            return False
    return True


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    open_question = "Please describe this image in detail."
    binary_question = "Is there a dog in the image?"
    inputs_a = to_device(prepare(processor, args.image_a, open_question), model.device)
    inputs_b = to_device(prepare(processor, args.image_b), model.device)
    inputs_binary = to_device(prepare(processor, args.image_a, binary_question), model.device)
    native_logits = first_logits(model, inputs_a)
    native_tokens = hf_generate(model, inputs_a, args.max_new_tokens)

    target_layers = qwen_target_layers(model)
    install_qwen_vhr(model, target_layers, aug_ratio=2.0, use_filter=True)
    try:
        set_qwen_vhr_phase(model, "off")
        patched_off_logits = first_logits(model, inputs_a)
        max_abs_logit_error = float((native_logits - patched_off_logits).abs().max().item())

        vanilla = generate_qwen(model, processor, inputs_a, "vanilla", args.max_new_tokens)

        set_ratio(model, 1.0)
        ratio_one = generate_qwen(model, processor, inputs_a, "vhr", args.max_new_tokens)

        set_ratio(model, 2.0)
        vhr_a = generate_qwen(model, processor, inputs_a, "vhr", args.max_new_tokens)
        vhr_b_after_a = generate_qwen(model, processor, inputs_b, "vhr", args.max_new_tokens)
        vhr_b_fresh = generate_qwen(model, processor, inputs_b, "vhr", args.max_new_tokens)
        sc_a = generate_qwen(model, processor, inputs_a, "scvhr", args.max_new_tokens)
        rr_open = generate_qwen(
            model,
            processor,
            inputs_a,
            qwen_rr_mode(open_question),
            args.max_new_tokens,
        )
        vhr_binary = generate_qwen(model, processor, inputs_binary, "vhr", args.max_new_tokens)
        rr_binary = generate_qwen(
            model,
            processor,
            inputs_binary,
            qwen_rr_mode(binary_question),
            args.max_new_tokens,
        )

        checks = {
            "patch_off_logits_within_1e-4": max_abs_logit_error <= 1e-4,
            "manual_vanilla_matches_hf_generate": vanilla.token_ids == native_tokens,
            "ratio_one_matches_vanilla": ratio_one.token_ids == vanilla.token_ids,
            "a_then_b_matches_fresh_b": vhr_b_after_a.token_ids == vhr_b_fresh.token_ids,
            "vhr_state_valid_a": state_is_valid(vhr_a, target_layers, model.config.num_attention_heads),
            "vhr_state_valid_b": state_is_valid(
                vhr_b_after_a,
                target_layers,
                model.config.num_attention_heads,
            ),
            "sc_forward_counts_valid": (
                sc_a.forward_counts["text"] == 1
                and sc_a.forward_counts["vhr"] == len(sc_a.token_ids)
                and sc_a.forward_counts["base"] == len(sc_a.token_ids)
            ),
            "rr_open_matches_sc": rr_open.token_ids == sc_a.token_ids,
            "rr_binary_matches_vhr": rr_binary.token_ids == vhr_binary.token_ids,
        }
        report = {
            "model_path": str(Path(args.model_path).resolve()),
            "dtype": str(model.dtype),
            "attention_implementation": model.config._attn_implementation,
            "target_layers": target_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "max_abs_logit_error": max_abs_logit_error,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "native_token_ids": native_tokens,
            "vanilla": vanilla.to_dict(),
            "ratio_one_vhr": ratio_one.to_dict(),
            "vhr_a": vhr_a.to_dict(),
            "vhr_b_after_a": vhr_b_after_a.to_dict(),
            "vhr_b_fresh": vhr_b_fresh.to_dict(),
            "sc_a": sc_a.to_dict(),
            "rr_open": rr_open.to_dict(),
            "vhr_binary": vhr_binary.to_dict(),
            "rr_binary": rr_binary.to_dict(),
        }
    finally:
        uninstall_qwen_vhr(model)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
