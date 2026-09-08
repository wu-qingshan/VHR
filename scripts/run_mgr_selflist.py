#!/usr/bin/env python3
"""Run the preregistered text-only self-listed entity extraction (stage 1)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_mgr import (
    SELFLIST_MAX_NEW_TOKENS,
    align_items,
    format_selflist_prompt,
    mention_to_dict,
    parse_selflist_output,
)


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


@torch.inference_mode()
def generate_selflist(model, processor, caption):
    prompt = format_selflist_prompt(caption)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    model.rope_deltas = None
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=SELFLIST_MAX_NEW_TOKENS,
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - start
    model.rope_deltas = None
    generated = output[0][inputs["input_ids"].shape[1] :].tolist()
    return {
        "prompt": prompt,
        "raw_text": processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ),
        "token_ids": generated,
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(model.device)
        / 1024**3
        if model.device.type == "cuda"
        else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--captions", required=True)
    parser.add_argument("--set_name", required=True, choices=("old", "vera"))
    parser.add_argument("--output_dir", default="mgr_verifier_results")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError("chunk_idx must be in [0, num_chunks)")

    captions = load_jsonl(args.captions)
    ids = [int(row["question_id"]) for row in captions]
    assigned = ids[args.chunk_idx :: args.num_chunks]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"selflist-{args.set_name}-chunk{args.chunk_idx:02d}.jsonl"
    completed = set()
    if output.exists():
        completed = {
            int(json.loads(line)["question_id"])
            for line in output.open()
            if line.strip()
        }

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    with output.open("a") as handle:
        for row in tqdm(
            (row for row in captions if int(row["question_id"]) in assigned),
            desc=f"selflist {args.set_name} {args.chunk_idx}",
            ncols=100,
        ):
            image_id = int(row["question_id"])
            if image_id in completed:
                continue
            caption = row["text"]
            result = generate_selflist(model, processor, caption)
            items, stats = parse_selflist_output(result["raw_text"])
            mentions, unaligned = align_items(caption, items)
            record = {
                "set": args.set_name,
                "question_id": image_id,
                "caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
                **result,
                "items": items,
                "parse_stats": stats,
                "mentions": [mention_to_dict(mention) for mention in mentions],
                "unaligned_items": unaligned,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    print(json.dumps({"set": args.set_name, "chunk": args.chunk_idx, "output": str(output)}))


if __name__ == "__main__":
    main()
