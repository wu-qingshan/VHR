#!/usr/bin/env python3
"""Run grounding + cropped-region verifier for every stage-1 inventory mention."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from qwen_object_verifier import score_object
from qwen_scvhr import generate_qwen
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_mgr import (
    GROUNDING_MAX_NEW_TOKENS,
    GROUNDING_PROMPT,
    crop_box,
    parse_grounding_box,
)


VERIFIER_CHECK_OBJECT = "person"


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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
    parser.add_argument("--tasks", default="mgr_verifier_results/region_tasks.json")
    parser.add_argument("--output_dir", default="mgr_verifier_results")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError("chunk_idx must be in [0, num_chunks)")

    tasks = json.load(open(args.tasks))
    images = sorted({int(row["image_id"]) for row in tasks})
    assigned = set(images[args.chunk_idx :: args.num_chunks])
    tasks = [row for row in tasks if int(row["image_id"]) in assigned]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"region-chunk{args.chunk_idx:02d}.jsonl"
    completed = set()
    if output.exists():
        completed = {
            json.loads(line)["query_id"]
            for line in output.open()
            if line.strip() and json.loads(line).get("query_type") == "grounding"
        }

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    first_image_path = Path(
        f"val2014/COCO_val2014_{int(tasks[0]['image_id']):012d}.jpg"
    )
    with Image.open(first_image_path) as check_image:
        pre_check = score_object(
            model, processor, check_image.convert("RGB"), VERIFIER_CHECK_OBJECT
        )

    with output.open("a") as handle:
        for task in tqdm(tasks, desc=f"region {args.chunk_idx}", ncols=100):
            if task["query_id"] in completed:
                continue
            image_path = Path(
                f"val2014/COCO_val2014_{int(task['image_id']):012d}.jpg"
            )
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            prompt = GROUNDING_PROMPT.format(phrase=task["object"])
            inputs = prepare(processor, image_path, prompt)
            result = generate_qwen(
                model,
                processor,
                inputs,
                mode="vanilla",
                max_new_tokens=GROUNDING_MAX_NEW_TOKENS,
            )
            model.rope_deltas = None
            box, ungrounded_reason = parse_grounding_box(result.text)
            region_score = None
            region_elapsed = None
            crop = None
            crop_reason = ungrounded_reason
            if box is not None:
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    crop, crop_reason = crop_box(box, rgb.width, rgb.height)
                    if crop is not None:
                        region = rgb.crop(crop)
                        region_result = score_object(
                            model, processor, region, task["object"]
                        )
                        region_score = region_result["score"]
                        region_elapsed = region_result["elapsed_seconds"]
            record = {
                "query_type": "grounding",
                "query_id": task["query_id"],
                "set": task["set"],
                "image_id": task["image_id"],
                "object": task["object"],
                "source": task["source"],
                "prompt": prompt,
                "raw_text": result.text,
                "token_ids": result.token_ids,
                "grounding_elapsed_seconds": result.elapsed_seconds,
                "box": box,
                "crop": crop,
                "ungrounded_reason": crop_reason,
                "region_score": region_score,
                "region_elapsed_seconds": region_elapsed,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

        with Image.open(first_image_path) as check_image:
            post_check = score_object(
                model, processor, check_image.convert("RGB"), VERIFIER_CHECK_OBJECT
            )
        handle.write(
            json.dumps(
                {
                    "query_type": "verifier_check_pre",
                    "object": VERIFIER_CHECK_OBJECT,
                    "image_id": int(tasks[0]["image_id"]),
                    "score": pre_check["score"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "query_type": "verifier_check_post",
                    "object": VERIFIER_CHECK_OBJECT,
                    "image_id": int(tasks[0]["image_id"]),
                    "score": post_check["score"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(
        json.dumps(
            {
                "chunk": args.chunk_idx,
                "tasks": len(tasks),
                "output": str(output),
                "verifier_pre_score": pre_check["score"],
                "verifier_post_score": post_check["score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
