#!/usr/bin/env python3
"""Run Qwen2.5-VL native grounding for the E0.3 twenty-image smoke."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from qwen_object_verifier import score_object
from qwen_scvhr import generate_qwen
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vera import sanitize_extraction


GROUNDING_PROMPT = "Point to the {phrase} in the image."
VERIFIER_CHECK_OBJECT = "person"


def sample_images(tasks, seed=2026090401, count=20):
    image_ids = sorted({task["image_id"] for task in tasks})
    return sorted(random.Random(seed).sample(image_ids, count))


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
    parser.add_argument("--tasks", default="qwen_vera_dev_results/tasks.json")
    parser.add_argument(
        "--extractions", default="qwen_vera_dev_results/source/extractions.json"
    )
    parser.add_argument("--output", default="mgr_ceiling_results/e03_grounding.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026090401)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    tasks = json.load(open(args.tasks))
    extractions = {row["image_id"]: row for row in json.load(open(args.extractions))}
    sample = sample_images(tasks, args.seed, args.count)
    for image_id in sample:
        if image_id not in extractions:
            raise KeyError(f"extraction missing for sampled image {image_id}")

    mentions = []
    for image_id in sample:
        extraction = sanitize_extraction(extractions[image_id])
        for candidate_index, candidate in enumerate(extraction["candidates"]):
            mentions.append(
                {
                    "mention_id": f"{image_id}:{candidate_index:03d}:{candidate['root_lemma']}",
                    "image_id": image_id,
                    "candidate_index": candidate_index,
                    "root_lemma": candidate["root_lemma"],
                    "phrase": candidate["phrase"],
                }
            )

    output = Path(args.output)
    if output.exists():
        completed = {
            json.loads(line)["mention_id"]
            for line in output.open()
            if line.strip() and json.loads(line).get("query_type") == "grounding"
        }
        if completed:
            raise FileExistsError(
                f"{output} already holds {len(completed)} grounding rows; "
                "refusing to append a second pass into the same file"
            )
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

    check_image = Image.open(
        f"val2014/COCO_val2014_{int(sample[0]):012d}.jpg"
    ).convert("RGB")
    pre_check = score_object(model, processor, check_image, VERIFIER_CHECK_OBJECT)

    with output.open("a") as handle:
        for row in tqdm(mentions, desc="MGR grounding smoke", ncols=100):
            if row["mention_id"] in completed:
                continue
            image_path = Path(f"val2014/COCO_val2014_{int(row['image_id']):012d}.jpg")
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            prompt = GROUNDING_PROMPT.format(phrase=row["phrase"])
            inputs = prepare(processor, image_path, prompt)
            result = generate_qwen(
                model,
                processor,
                inputs,
                mode="vanilla",
                max_new_tokens=args.max_new_tokens,
            )
            model.rope_deltas = None
            record = {
                **row,
                "query_type": "grounding",
                "prompt": prompt,
                "raw_text": result.text,
                "token_ids": result.token_ids,
                "terminated": result.terminated,
                "elapsed_seconds": result.elapsed_seconds,
                "peak_allocated_gib": result.peak_allocated_gib,
                "image_grid_thw": inputs["image_grid_thw"][0].detach().cpu().tolist(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

        post_check = score_object(
            model, processor, check_image, VERIFIER_CHECK_OBJECT
        )
        handle.write(
            json.dumps(
                {
                    "query_type": "verifier_check_pre",
                    "object": VERIFIER_CHECK_OBJECT,
                    "image_id": int(sample[0]),
                    "score": pre_check["score"],
                    "yes_logit": pre_check["yes_logit"],
                    "no_logit": pre_check["no_logit"],
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
                    "image_id": int(sample[0]),
                    "score": post_check["score"],
                    "yes_logit": post_check["yes_logit"],
                    "no_logit": post_check["no_logit"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(
        json.dumps(
            {
                "images": len(sample),
                "mentions": len(mentions),
                "output": str(output),
                "verifier_pre_score": pre_check["score"],
                "verifier_post_score": post_check["score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
