#!/usr/bin/env python3
"""Stage-1 engineering smoke: determinism, order independence, state hygiene."""

import argparse
import hashlib
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_mgr import (
    GROUNDING_MAX_NEW_TOKENS,
    GROUNDING_PROMPT,
    align_items,
    crop_box,
    format_selflist_prompt,
    parse_grounding_box,
    parse_selflist_output,
)
from run_mgr_selflist import generate_selflist


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def ground(model, processor, image_path, phrase):
    inputs = prepare(processor, image_path, GROUNDING_PROMPT.format(phrase=phrase))
    result = generate_qwen(
        model,
        processor,
        inputs,
        mode="vanilla",
        max_new_tokens=GROUNDING_MAX_NEW_TOKENS,
    )
    model.rope_deltas = None
    return result


def atomic_json_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--captions", default="qwen_dev_results/vanilla.json")
    parser.add_argument("--output", default="mgr_verifier_results/engineering_smoke.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_captions", type=int, default=5)
    args = parser.parse_args()

    captions = load_jsonl(args.captions)[: args.num_captions]
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    checks = {}
    details = {}

    # E1: self-list determinism.
    first_pass = [generate_selflist(model, processor, row["text"]) for row in captions]
    second_pass = [generate_selflist(model, processor, row["text"]) for row in captions]
    checks["e1_selflist_determinism"] = all(
        a["token_ids"] == b["token_ids"] for a, b in zip(first_pass, second_pass)
    )
    details["e1"] = {
        str(int(row["question_id"])): {"tokens": len(a["token_ids"])}
        for row, a in zip(captions, first_pass)
    }

    # Parse and align once for downstream checks.
    parsed = []
    for row, result in zip(captions, first_pass):
        items, _ = parse_selflist_output(result["raw_text"])
        mentions, _ = align_items(row["text"], items)
        parsed.append((row, result, mentions))

    # E6: span integrity.
    span_ok = True
    span_bad = []
    for row, _, mentions in parsed:
        for mention in mentions:
            for occurrence in mention.occurrences:
                text = row["text"][occurrence.start_char : occurrence.end_char]
                if text.lower() != mention.mention_key:
                    span_ok = False
                    span_bad.append(
                        {
                            "image_id": row["question_id"],
                            "mention": mention.mention_key,
                            "span_text": text,
                        }
                    )
    checks["e6_span_integrity"] = span_ok
    details["e6"] = {"bad_spans": span_bad[:5], "total": span_bad}

    smoke_image = Path(
        f"val2014/COCO_val2014_{int(captions[0]['question_id']):012d}.jpg"
    )
    other_image = Path(
        f"val2014/COCO_val2014_{int(captions[1]['question_id']):012d}.jpg"
    )
    probe_objects = [
        mention.mention for mention in parsed[0][2]
    ][:2] or ["person"]

    # E2: whole verifier determinism.
    with Image.open(smoke_image) as image:
        rgb = image.convert("RGB")
        whole_first = {
            obj: score_object(model, processor, rgb, obj) for obj in probe_objects
        }
        whole_second = {
            obj: score_object(model, processor, rgb, obj) for obj in probe_objects
        }
    checks["e2_whole_verifier_determinism"] = all(
        abs(whole_first[obj]["score"] - whole_second[obj]["score"]) <= 1e-6
        for obj in probe_objects
    )

    # E3: grounding determinism (smoke image and other image).
    ground_a = ground(model, processor, smoke_image, probe_objects[0])
    ground_b = ground(model, processor, smoke_image, probe_objects[0])
    ground_other = ground(model, processor, other_image, probe_objects[0])
    checks["e3_grounding_determinism"] = (
        ground_a.text == ground_b.text and ground_a.token_ids == ground_b.token_ids
    )

    # Region verifier determinism on a real crop.
    box, reason = parse_grounding_box(ground_a.text)
    region_scores = []
    if box is not None:
        with Image.open(smoke_image) as image:
            rgb = image.convert("RGB")
            crop, crop_reason = crop_box(box, rgb.width, rgb.height)
        if crop is not None:
            for _ in range(2):
                region = rgb.crop(crop)
                region_result = score_object(
                    model, processor, region, probe_objects[0]
                )
                region_scores.append(region_result["score"])
    checks["e2_region_verifier_determinism"] = (
        len(region_scores) == 2 and abs(region_scores[0] - region_scores[1]) <= 1e-6
    )

    # E4: order independence for the whole verifier.
    with Image.open(smoke_image) as image:
        rgb = image.convert("RGB")
        before = score_object(model, processor, rgb, probe_objects[0])["score"]
    ground(model, processor, other_image, probe_objects[0])
    with Image.open(smoke_image) as image:
        rgb = image.convert("RGB")
        after_grounding = score_object(model, processor, rgb, probe_objects[0])["score"]
    box_other, _ = parse_grounding_box(ground_other.text)
    if box_other is not None:
        with Image.open(other_image) as image:
            other_rgb = image.convert("RGB")
            crop_other, _ = crop_box(box_other, other_rgb.width, other_rgb.height)
        if crop_other is not None:
            score_object(
                model, processor, other_rgb.crop(crop_other), probe_objects[0]
            )
    with Image.open(smoke_image) as image:
        rgb = image.convert("RGB")
        after_region = score_object(model, processor, rgb, probe_objects[0])["score"]
    checks["e4_order_independence"] = (
        abs(before - after_grounding) <= 1e-6 and abs(before - after_region) <= 1e-6
    )

    # E5: rope_deltas hygiene after each operation type.
    checks["e5_rope_deltas_none"] = model.rope_deltas is None
    with Image.open(smoke_image) as image:
        rgb = image.convert("RGB")
        leak_pre = score_object(model, processor, rgb, probe_objects[0])["score"]
    generate_selflist(model, processor, captions[0]["text"])
    ground(model, processor, other_image, probe_objects[0])
    with Image.open(smoke_image) as image:
        rgb = image.convert("RGB")
        leak_post = score_object(model, processor, rgb, probe_objects[0])["score"]
    checks["e5_verifier_leak"] = leak_pre == leak_post and model.rope_deltas is None
    details["e5"] = {"pre": leak_pre, "post": leak_post}

    details["e2"] = {
        "objects": probe_objects,
        "scores": {obj: whole_first[obj]["score"] for obj in probe_objects},
        "region_scores": region_scores,
    }
    details["e3"] = {"raw_text": ground_a.text[:200]}
    details["e4"] = {
        "before": before,
        "after_grounding": after_grounding,
        "after_region": after_region,
    }
    details["e7_input_hashes"] = {
        "captions": sha256_file(args.captions),
        "model_config": sha256_file(Path(args.model_path) / "config.json"),
    }

    report = {
        "model_path": args.model_path,
        "checks": checks,
        "all_passed": all(checks.values()),
        **details,
    }
    atomic_json_new(args.output, report)
    print(json.dumps({key: report[key] for key in ("checks", "all_passed")}, indent=2))


if __name__ == "__main__":
    main()
