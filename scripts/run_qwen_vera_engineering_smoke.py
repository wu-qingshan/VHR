#!/usr/bin/env python3
"""Reproducible HF/ratio/order engineering smoke for VERA revision decoding."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_scvhr import generate_qwen, qwen_target_layers
from qwen_vhr import install_qwen_vhr, uninstall_qwen_vhr
from scripts.run_qwen_vera_revisions import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--tasks", default="qwen_vera_smoke_v2_results/tasks.json")
    parser.add_argument("--task_index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="qwen_vera_engineering_results/smoke.json")
    args = parser.parse_args()

    task = json.load(open(args.tasks))[args.task_index]
    image_path = Path(f"val2014/COCO_val2014_{int(task['image_id']):012d}.jpg")
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    inputs = prepare(processor, image_path, task["prompt"])

    gpu_inputs = {key: value.to(args.device) for key, value in inputs.items()}
    model.rope_deltas = None
    with torch.inference_mode():
        generated = model.generate(
            **gpu_inputs,
            max_new_tokens=128,
            do_sample=False,
        )
    prompt_length = inputs["input_ids"].shape[1]
    hf_ids = generated[0, prompt_length:].tolist()

    install_qwen_vhr(
        model, qwen_target_layers(model), aug_ratio=1.0, use_filter=True
    )
    ratio_one = generate_qwen(
        model, processor, inputs, mode="vhr", max_new_tokens=128
    )
    uninstall_qwen_vhr(model)

    install_qwen_vhr(
        model, qwen_target_layers(model), aug_ratio=2.0, use_filter=True
    )
    base_a = generate_qwen(
        model, processor, inputs, mode="vanilla", max_new_tokens=128
    )
    vhr_a = generate_qwen(model, processor, inputs, mode="vhr", max_new_tokens=128)
    vhr_b = generate_qwen(model, processor, inputs, mode="vhr", max_new_tokens=128)
    base_b = generate_qwen(
        model, processor, inputs, mode="vanilla", max_new_tokens=128
    )
    uninstall_qwen_vhr(model)

    checks = {
        "base_matches_hf_tokens": base_a.token_ids == hf_ids,
        "ratio_one_vhr_matches_hf_tokens": ratio_one.token_ids == hf_ids,
        "base_repeat_tokens_equal": base_a.token_ids == base_b.token_ids,
        "vhr_repeat_tokens_equal": vhr_a.token_ids == vhr_b.token_ids,
        "base_repeat_text_equal": base_a.text == base_b.text,
        "vhr_repeat_text_equal": vhr_a.text == vhr_b.text,
        "base_updates_every_generated_token": base_a.forward_counts["base"]
        == len(base_a.token_ids),
    }
    report = {
        "task": task,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "hf": {
            "token_ids": hf_ids,
            "text": processor.decode(
                hf_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip(),
        },
        "ratio_one": ratio_one.to_dict(),
        "base_a": base_a.to_dict(),
        "vhr_a": vhr_a.to_dict(),
        "vhr_b": vhr_b.to_dict(),
        "base_b": base_b.to_dict(),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "checks": checks,
                "all_checks_passed": report["all_checks_passed"],
                "hf_text": report["hf"]["text"],
                "vhr_text": vhr_a.text,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
