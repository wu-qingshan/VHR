#!/usr/bin/env python3
"""Build CHAIR-labeled unique object queries from frozen vanilla captions."""

import argparse
import json
import os
from pathlib import Path

from datasets import CHAIRBench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", default="experiment/qwen_dev_chair_ids.json")
    parser.add_argument("--captions", default="qwen_dev_results/vanilla.json")
    parser.add_argument("--output", default="qwen_object_evidence_results/queries.json")
    args = parser.parse_args()

    ids = [int(value) for value in json.load(open(args.ids_file))]
    rows = [json.loads(line) for line in open(args.captions) if line.strip()]
    if [int(row["question_id"]) for row in rows] != ids:
        raise ValueError("vanilla captions do not match frozen dev IDs")
    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    queries = []
    for row in rows:
        image_id = int(row["question_id"])
        ground_truth, node_words = dataset.eval_hal_words(image_id, row["text"])
        seen = set()
        for object_name in node_words:
            if object_name in seen:
                continue
            seen.add(object_name)
            queries.append(
                {
                    "query_id": f"{image_id}:{object_name}",
                    "image_id": image_id,
                    "object": object_name,
                    "label": int(object_name in ground_truth),
                }
            )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(queries, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "queries": len(queries),
                "positive": sum(row["label"] for row in queries),
                "negative": sum(1 - row["label"] for row in queries),
                "images": len(set(row["image_id"] for row in queries)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
