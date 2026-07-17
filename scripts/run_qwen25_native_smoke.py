#!/usr/bin/env python3
"""Run the preregistered native Qwen2.5-VL engineering smoke.

This script does not import or modify the historical VHR generation monkeypatch.
It verifies the official chat template, dynamic image-token accounting, mRoPE
inputs, deterministic greedy generation, and termination on two real images.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--image", action="append", required=True, help="repeat exactly twice")
    parser.add_argument("--prompt", default="Describe this image in one concise sentence.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output", default="qwen_native_smoke_results/audit.json")
    args = parser.parse_args()
    if len(args.image) != 2:
        parser.error("--image must be supplied exactly twice")
    return args


def prepare_inputs(processor, image_path, prompt):
    absolute_image = str(Path(image_path).resolve())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": absolute_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return messages, text, inputs


def audit_inputs(model, inputs):
    input_ids = inputs["input_ids"]
    image_positions = (input_ids[0] == model.config.image_token_id).nonzero().flatten()
    vision_starts = (input_ids[0] == model.config.vision_start_token_id).nonzero().flatten()
    vision_ends = (input_ids[0] == model.config.vision_end_token_id).nonzero().flatten()
    if image_positions.numel() == 0:
        raise AssertionError("processor produced no image tokens")
    if vision_starts.numel() != 1 or vision_ends.numel() != 1:
        raise AssertionError(
            f"expected one vision span, got starts={vision_starts.tolist()} ends={vision_ends.tolist()}"
        )
    if not (vision_starts.item() < image_positions[0].item() <= image_positions[-1].item() < vision_ends.item()):
        raise AssertionError("image tokens are not enclosed by the vision boundary tokens")

    grid = inputs["image_grid_thw"][0].tolist()
    merge_size = int(model.config.vision_config.spatial_merge_size)
    expected_image_tokens = int(grid[0] * grid[1] * grid[2] // (merge_size**2))
    actual_image_tokens = int(image_positions.numel())
    if actual_image_tokens != expected_image_tokens:
        raise AssertionError(
            f"image-token mismatch: actual={actual_image_tokens} expected={expected_image_tokens} grid={grid}"
        )

    return {
        "input_ids_shape": list(input_ids.shape),
        "attention_mask_shape": list(inputs["attention_mask"].shape),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "image_grid_thw": grid,
        "spatial_merge_size": merge_size,
        "actual_image_tokens": actual_image_tokens,
        "expected_image_tokens": expected_image_tokens,
        "vision_span": [int(vision_starts.item()), int(vision_ends.item()) + 1],
        "image_token_span": [int(image_positions[0].item()), int(image_positions[-1].item()) + 1],
    }


@torch.inference_mode()
def generate_once(model, processor, inputs, max_new_tokens):
    device_inputs = {key: value.to(model.device) for key, value in inputs.items()}
    torch.cuda.reset_peak_memory_stats(model.device)
    start = time.perf_counter()
    generated = model.generate(
        **device_inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - start
    prompt_length = device_inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return {
        "token_ids": new_tokens[0].cpu().tolist(),
        "text": text,
        "new_token_count": int(new_tokens.shape[1]),
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(model.device) / 1024**3,
        "terminated_before_limit": int(new_tokens.shape[1]) < max_new_tokens,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen2.5-VL-7B smoke")

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    report = {
        "model_path": str(Path(args.model_path).resolve()),
        "expected_revision": DEFAULT_REVISION,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "dtype": str(model.dtype),
        "attention_implementation": model.config._attn_implementation,
        "device": str(model.device),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "images": [],
    }

    prepared = []
    for image_path in args.image:
        _, chat_text, inputs = prepare_inputs(processor, image_path, args.prompt)
        audit = audit_inputs(model, inputs)
        prepared.append(inputs)
        report["images"].append(
            {
                "path": str(Path(image_path).resolve()),
                "chat_text": chat_text,
                "input_audit": audit,
            }
        )

    first_a = generate_once(model, processor, prepared[0], args.max_new_tokens)
    first_b = generate_once(model, processor, prepared[0], args.max_new_tokens)
    second = generate_once(model, processor, prepared[1], args.max_new_tokens)
    report["images"][0]["generation_0"] = first_a
    report["images"][0]["generation_1"] = first_b
    report["images"][1]["generation_0"] = second
    report["deterministic_first_image"] = first_a["token_ids"] == first_b["token_ids"]
    report["dynamic_image_token_counts"] = (
        report["images"][0]["input_audit"]["actual_image_tokens"]
        != report["images"][1]["input_audit"]["actual_image_tokens"]
    )
    report["all_nonempty"] = all(
        entry.get("generation_0", {}).get("new_token_count", 0) > 0 for entry in report["images"]
    )
    report["all_terminated_before_limit"] = all(
        entry.get("generation_0", {}).get("terminated_before_limit", False)
        for entry in report["images"]
    )
    report["all_checks_passed"] = all(
        [
            report["deterministic_first_image"],
            report["dynamic_image_token_counts"],
            report["all_nonempty"],
            report["all_terminated_before_limit"],
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
