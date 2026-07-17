#!/usr/bin/env python3
"""Engineering smoke for generic and pathological SEAR verifier phrases."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_object_verifier import score_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--queries", default="qwen_sear_dev_results/queries.json")
    parser.add_argument(
        "--reference", default="qwen_object_evidence_results/verifier.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="qwen_sear_dev_results/engineering_smoke.json")
    args = parser.parse_args()

    queries = json.load(open(args.queries))
    references = [json.loads(line) for line in open(args.reference) if line.strip()]
    reference_by_key = {
        (int(row["image_id"]), row["object"]): row for row in references
    }
    shared = next(
        query
        for query in queries
        if (int(query["image_id"]), query["object"]) in reference_by_key
    )
    longest = max(queries, key=lambda row: len(row["object"].split()))

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    def run(query):
        image_id = int(query["image_id"])
        image = Image.open(
            f"val2014/COCO_val2014_{image_id:012d}.jpg"
        ).convert("RGB")
        return score_object(model, processor, image, query["object"])

    shared_result = run(shared)
    longest_first = run(longest)
    longest_second = run(longest)
    reference_score = float(
        reference_by_key[(int(shared["image_id"]), shared["object"])]["score"]
    )
    checks = {
        "shared_query_matches_frozen_verifier": abs(
            shared_result["score"] - reference_score
        )
        <= 1e-6,
        "longest_query_is_finite": all(
            math.isfinite(float(longest_first[key]))
            for key in ("yes_logit", "no_logit", "score")
        ),
        "longest_query_repeat_within_1e_6": abs(
            longest_first["score"] - longest_second["score"]
        )
        <= 1e-6,
    }
    report = {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "shared_query": shared,
        "shared_reference_score": reference_score,
        "shared_result": shared_result,
        "longest_query": {
            **longest,
            "word_count": len(longest["object"].split()),
        },
        "longest_first": longest_first,
        "longest_second": longest_second,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "checks": checks,
                "all_checks_passed": report["all_checks_passed"],
                "shared_query": shared["query_id"],
                "longest_query": longest["query_id"],
                "longest_query_words": len(longest["object"].split()),
            },
            indent=2,
        )
    )
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
