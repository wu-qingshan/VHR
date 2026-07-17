#!/usr/bin/env python3
"""Run fixed Qwen object verifier queries for one GPU chunk."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_object_verifier import score_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--queries", default="qwen_object_evidence_results/queries.json")
    parser.add_argument("--output_dir", default="qwen_object_evidence_results")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    all_queries = json.load(open(args.queries))
    image_order = []
    seen_images = set()
    for query in all_queries:
        if query["image_id"] not in seen_images:
            seen_images.add(query["image_id"])
            image_order.append(query["image_id"])
    assigned_images = set(image_order[args.chunk_idx :: args.num_chunks])
    queries = [query for query in all_queries if query["image_id"] in assigned_images]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"chunk{args.chunk_idx:02d}-verifier.json"
    completed = set()
    if output.exists():
        completed = {json.loads(line)["query_id"] for line in output.open() if line.strip()}

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    image_id = None
    image = None
    with output.open("a") as handle:
        for query in tqdm(queries, desc=f"Verifier chunk {args.chunk_idx}", ncols=100):
            if query["query_id"] in completed:
                continue
            if image_id != query["image_id"]:
                image_id = query["image_id"]
                image = Image.open(f"val2014/COCO_val2014_{image_id:012d}.jpg").convert("RGB")
            result = score_object(model, processor, image, query["object"])
            row = {**query, **result}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
