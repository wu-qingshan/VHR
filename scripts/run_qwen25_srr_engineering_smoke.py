#!/usr/bin/env python3
"""GPU engineering smoke for irreversible stability-risk rollback."""

import argparse
import json
import os
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_scvhr import generate_qwen, qwen_target_layers
from qwen_vhr import install_qwen_vhr, uninstall_qwen_vhr
from scripts.analyze_caption_quality import exact_cycle


PROMPT = "Please describe this image in detail."


def prepare(processor, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(image_path).resolve())},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    return processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="qwen_srr_smoke_results/engineering_audit.json")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    install_qwen_vhr(model, qwen_target_layers(model), aug_ratio=2.0, use_filter=True)
    inputs = prepare(processor, args.image)
    try:
        sc_short = generate_qwen(model, processor, inputs, "scvhr", 8)
        srr_short = generate_qwen(model, processor, inputs, "srrscvhr", 8)
        srr_full = generate_qwen(model, processor, inputs, "srrscvhr", 512)
        srr_repeat = generate_qwen(model, processor, inputs, "srrscvhr", 512)
    finally:
        uninstall_qwen_vhr(model)

    route = srr_full.routing_stats
    checks = {
        "no_risk_prefix_matches_sc": srr_short.token_ids == sc_short.token_ids,
        "rollback_triggered": route["rollback_triggered"] is True,
        "rollback_step_matches_vhr_forward_count": (
            route["rollback_step"] is not None
            and srr_full.forward_counts["vhr"] == route["rollback_step"] + 1
        ),
        "vhr_forward_stops_after_rollback": (
            srr_full.forward_counts["vhr"] < srr_full.forward_counts["base"]
        ),
        "base_cache_covers_generation": srr_full.forward_counts["base"] == len(srr_full.token_ids),
        "output_has_no_exact_cycle": exact_cycle(srr_full.text, 16, 3) is None,
        "repeat_run_identical": srr_full.token_ids == srr_repeat.token_ids,
    }
    report = {
        "image": str(Path(args.image).resolve()),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "sc_short": sc_short.to_dict(),
        "srr_short": srr_short.to_dict(),
        "srr_full": srr_full.to_dict(),
        "srr_repeat": srr_repeat.to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"checks": checks, "srr_full": srr_full.to_dict()}, indent=2, ensure_ascii=False))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
