#!/usr/bin/env python3
"""Engineering smoke for deterministic Qwen object Yes/No logits."""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_object_verifier import score_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--dog_image", default="val2014/COCO_val2014_000000000042.jpg")
    parser.add_argument("--giraffe_image", default="val2014/COCO_val2014_000000000136.jpg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="qwen_object_evidence_results/smoke.json")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    dog_image = Image.open(args.dog_image).convert("RGB")
    giraffe_image = Image.open(args.giraffe_image).convert("RGB")

    dog_a = score_object(model, processor, dog_image, "dog")
    elephant = score_object(model, processor, dog_image, "elephant")
    giraffe_a = score_object(model, processor, giraffe_image, "giraffe")
    dog_b = score_object(model, processor, dog_image, "dog")
    giraffe_b = score_object(model, processor, giraffe_image, "giraffe")
    checks = {
        "dog_score_above_elephant": dog_a["score"] > elephant["score"],
        "dog_repeat_within_1e_6": abs(dog_a["score"] - dog_b["score"]) <= 1e-6,
        "a_then_b_matches_fresh_b": abs(giraffe_a["score"] - giraffe_b["score"]) <= 1e-6,
        "dynamic_grid": dog_a["image_grid_thw"] != giraffe_a["image_grid_thw"],
    }
    report = {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "dog": dog_a,
        "elephant": elephant,
        "giraffe": giraffe_a,
        "dog_repeat": dog_b,
        "giraffe_repeat": giraffe_b,
    }
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
