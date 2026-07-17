#!/usr/bin/env python3
"""Run frozen Qwen vanilla/VHR/SC-VHR CHAIR generation for one GPU chunk."""

import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_scvhr import generate_qwen, qwen_target_layers
from qwen_vhr import install_qwen_vhr, uninstall_qwen_vhr


PROMPT = "Please describe this image in detail."
VALID_METHODS = ("vanilla", "vhr", "scvhr")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--ids_file", required=True)
    parser.add_argument("--output_dir", default="qwen_dev_results")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", default=",".join(VALID_METHODS))
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not 0 <= args.chunk_idx < args.num_chunks:
        parser.error("chunk_idx must be in [0, num_chunks)")
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    unknown = set(methods) - set(VALID_METHODS)
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}")
    args.methods = methods
    return args


def prepare(processor, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": PROMPT},
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


def existing_ids(path):
    if not path.exists():
        return set()
    output = set()
    with path.open() as handle:
        for line in handle:
            if line.strip():
                output.add(int(json.loads(line)["question_id"]))
    return output


def main():
    args = parse_args()
    all_ids = [int(value) for value in json.load(open(args.ids_file))]
    chunk_ids = all_ids[args.chunk_idx :: args.num_chunks]
    if args.limit is not None:
        chunk_ids = chunk_ids[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        method: output_dir / f"chunk{args.chunk_idx:02d}-{method}.json"
        for method in args.methods
    }
    completed = {method: existing_ids(path) for method, path in paths.items()}

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    target_layers = qwen_target_layers(model)
    install_qwen_vhr(model, target_layers, aug_ratio=2.0, use_filter=True)

    handles = {method: path.open("a") for method, path in paths.items()}
    try:
        for image_id in tqdm(chunk_ids, desc=f"Qwen chunk {args.chunk_idx}", ncols=100):
            pending = [method for method in args.methods if image_id not in completed[method]]
            if not pending:
                continue
            image_path = Path(f"val2014/COCO_val2014_{image_id:012d}.jpg")
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            inputs = prepare(processor, image_path)
            for method in pending:
                result = generate_qwen(
                    model,
                    processor,
                    inputs,
                    mode=method,
                    max_new_tokens=args.max_new_tokens,
                    alpha=0.5,
                    beta=0.1,
                )
                row = {
                    "question_id": image_id,
                    "text": result.text,
                    "question": PROMPT,
                    "method": method,
                    "ans_conf": f"{100 * result.first_token_confidence:.2f}",
                    "token_count": len(result.token_ids),
                    "terminated": result.terminated,
                    "elapsed_seconds": result.elapsed_seconds,
                    "peak_allocated_gib": result.peak_allocated_gib,
                    "forward_counts": result.forward_counts,
                }
                handles[method].write(json.dumps(row, ensure_ascii=False) + "\n")
                handles[method].flush()
                completed[method].add(image_id)
    finally:
        for handle in handles.values():
            handle.close()
        uninstall_qwen_vhr(model)


if __name__ == "__main__":
    main()
