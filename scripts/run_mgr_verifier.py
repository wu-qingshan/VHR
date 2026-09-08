#!/usr/bin/env python3
"""Run whole-image verifier queries for new stage-1 inventory objects."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_object_verifier import score_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument(
        "--queries", default="mgr_verifier_results/whole_queries.json"
    )
    parser.add_argument("--output_dir", default="mgr_verifier_results")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError("chunk_idx must be in [0, num_chunks)")

    payload = json.load(open(args.queries))
    queries = [row for row in payload["queries"] if row["reused_query_id"] is None]
    images = sorted({int(row["image_id"]) for row in queries})
    assigned = set(images[args.chunk_idx :: args.num_chunks])
    queries = [row for row in queries if int(row["image_id"]) in assigned]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"whole-verifier-chunk{args.chunk_idx:02d}.jsonl"
    completed = set()
    if output.exists():
        completed = {
            json.loads(line)["query_id"]
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
        for query in tqdm(queries, desc=f"whole verifier {args.chunk_idx}", ncols=100):
            if query["query_id"] in completed:
                continue
            image_path = Path(
                f"val2014/COCO_val2014_{int(query['image_id']):012d}.jpg"
            )
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                result = score_object(
                    model, processor, image.convert("RGB"), query["object"]
                )
            handle.write(
                json.dumps(
                    {
                        "query_id": query["query_id"],
                        "set": query["set"],
                        "image_id": query["image_id"],
                        "object": query["object"],
                        "score": result["score"],
                        "yes_logit": result["yes_logit"],
                        "no_logit": result["no_logit"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "peak_allocated_gib": result["peak_allocated_gib"],
                        "image_grid_thw": result["image_grid_thw"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
    print(json.dumps({"new_queries": len(queries), "output": str(output)}))


if __name__ == "__main__":
    main()
