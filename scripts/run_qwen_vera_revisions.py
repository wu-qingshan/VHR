#!/usr/bin/env python3
"""Generate frozen base and VHR local-revision candidates for one GPU chunk."""

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

from qwen_scvhr import generate_qwen, qwen_target_layers
from qwen_vhr import install_qwen_vhr, uninstall_qwen_vhr


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
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--order", default="base,vhr")
    args = parser.parse_args()
    if args.max_new_tokens != 128:
        raise ValueError("VERA max_new_tokens is preregistered at 128")
    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError("chunk_idx must be in [0, num_chunks)")
    order = tuple(value.strip() for value in args.order.split(",") if value.strip())
    if len(order) != 2 or set(order) != {"base", "vhr"}:
        raise ValueError("order must contain base and vhr exactly once")

    all_tasks = json.load(open(args.tasks))
    image_order = []
    seen_images = set()
    for task in all_tasks:
        if task["image_id"] not in seen_images:
            seen_images.add(task["image_id"])
            image_order.append(task["image_id"])
    assigned_images = set(image_order[args.chunk_idx :: args.num_chunks])
    tasks = [task for task in all_tasks if task["image_id"] in assigned_images]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"chunk{args.chunk_idx:02d}-revisions.json"
    completed = set()
    if output.exists():
        completed = {
            json.loads(line)["revision_id"] for line in output.open() if line.strip()
        }

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    install_qwen_vhr(
        model, qwen_target_layers(model), aug_ratio=2.0, use_filter=True
    )
    try:
        with output.open("a") as handle:
            for task in tqdm(tasks, desc=f"VERA revisions {args.chunk_idx}", ncols=100):
                if task["revision_id"] in completed:
                    continue
                image_path = Path(
                    f"val2014/COCO_val2014_{int(task['image_id']):012d}.jpg"
                )
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                candidates = {}
                for label in order:
                    mode = "vanilla" if label == "base" else "vhr"
                    inputs = prepare(processor, image_path, task["prompt"])
                    result = generate_qwen(
                        model,
                        processor,
                        inputs,
                        mode=mode,
                        max_new_tokens=args.max_new_tokens,
                    )
                    candidates[label] = {
                        "text": result.text,
                        "token_ids": result.token_ids,
                        "terminated": result.terminated,
                        "elapsed_seconds": result.elapsed_seconds,
                        "peak_allocated_gib": result.peak_allocated_gib,
                        "first_token_confidence": result.first_token_confidence,
                        "forward_counts": result.forward_counts,
                    }
                row = {**task, "candidates": candidates}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed.add(task["revision_id"])
    finally:
        uninstall_qwen_vhr(model)


if __name__ == "__main__":
    main()
