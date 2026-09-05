#!/usr/bin/env python3
"""Generate base-only VERA-style rewrite candidates for MGR ceiling operator C."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_scvhr import generate_qwen


def prepare(processor, image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--tasks", default="mgr_ceiling_results/operatorC_tasks.json")
    parser.add_argument("--output", default="mgr_ceiling_results/rewrite_candidates.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_new_tokens != 128:
        raise ValueError("operator C max_new_tokens is preregistered at 128")

    tasks = json.load(open(args.tasks))
    output = Path(args.output)
    if output.exists():
        completed = {
            json.loads(line)["revision_id"]
            for line in output.open()
            if line.strip()
        }
    else:
        completed = set()

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    with output.open("a") as handle:
        for task in tqdm(tasks, desc="MGR ceiling rewrites", ncols=100):
            if task["revision_id"] in completed:
                continue
            image_path = Path(
                f"val2014/COCO_val2014_{int(task['image_id']):012d}.jpg"
            )
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            inputs = prepare(processor, image_path, task["prompt"])
            result = generate_qwen(
                model,
                processor,
                inputs,
                mode="vanilla",
                max_new_tokens=args.max_new_tokens,
            )
            row = {
                **task,
                "candidate": {
                    "text": result.text,
                    "token_ids": result.token_ids,
                    "terminated": result.terminated,
                    "elapsed_seconds": result.elapsed_seconds,
                    "peak_allocated_gib": result.peak_allocated_gib,
                    "first_token_confidence": result.first_token_confidence,
                    "forward_counts": result.forward_counts,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed.add(task["revision_id"])
    print(json.dumps({"tasks": len(tasks), "output": str(output)}))


if __name__ == "__main__":
    main()
