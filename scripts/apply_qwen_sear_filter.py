#!/usr/bin/env python3
"""Apply frozen SEAR sentence filtering to verifier-scored Qwen dev captions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_sear import filter_caption


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", default="qwen_dev_results/vanilla.json")
    parser.add_argument("--extractions", default="qwen_sear_dev_results/extractions.json")
    parser.add_argument("--verifier", default="qwen_sear_dev_results/verifier.json")
    parser.add_argument("--output", default="qwen_sear_dev_results/sear.json")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    if args.threshold != 0.1:
        raise ValueError("SEAR threshold is preregistered and frozen at 0.1")

    captions = load_jsonl(args.captions)
    extractions = json.load(open(args.extractions))
    verifier = load_jsonl(args.verifier)
    if [int(row["question_id"]) for row in captions] != [
        int(row["image_id"]) for row in extractions
    ]:
        raise ValueError("caption/extraction image order mismatch")
    by_query = {}
    verifier_elapsed = {}
    verifier_peak = {}
    for row in verifier:
        query_id = row["query_id"]
        if query_id in by_query:
            raise ValueError(f"duplicate verifier result {query_id}")
        by_query[query_id] = float(row["score"])
        image_id = int(row["image_id"])
        verifier_elapsed[image_id] = verifier_elapsed.get(image_id, 0.0) + float(
            row["elapsed_seconds"]
        )
        verifier_peak[image_id] = max(
            verifier_peak.get(image_id, 0.0), float(row["peak_allocated_gib"])
        )
    expected_queries = {
        candidate["query_id"]
        for extraction in extractions
        for candidate in extraction["candidates"]
    }
    if set(by_query) != expected_queries:
        raise ValueError(
            f"verifier query mismatch: missing={len(expected_queries - set(by_query))} "
            f"extra={len(set(by_query) - expected_queries)}"
        )

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for caption, extraction in zip(captions, extractions):
            if sha256_text(caption["text"]) != extraction["text_sha256"]:
                raise ValueError(f"caption text changed for {caption['question_id']}")
            image_id = int(caption["question_id"])
            filtered, stats = filter_caption(
                caption["text"], extraction, by_query, threshold=args.threshold
            )
            parse_seconds = float(extraction["extraction_elapsed_seconds"])
            verify_seconds = verifier_elapsed.get(image_id, 0.0)
            row = {
                **caption,
                "text": filtered,
                "method": "sear",
                "source_method": caption.get("method"),
                "source_text_sha256": extraction["text_sha256"],
                "elapsed_seconds": float(caption["elapsed_seconds"])
                + parse_seconds
                + verify_seconds
                + stats["filter_elapsed_seconds"],
                "peak_allocated_gib": max(
                    float(caption["peak_allocated_gib"]), verifier_peak.get(image_id, 0.0)
                ),
                "sear_stats": {
                    **stats,
                    "threshold": args.threshold,
                    "candidate_count": len(extraction["candidates"]),
                    "source_sentence_count": len(extraction["sentences"]),
                    "retained_sentence_count": len(stats["retained_sentence_indices"]),
                    "extraction_elapsed_seconds": parse_seconds,
                    "verifier_elapsed_seconds": verify_seconds,
                    "sear_only_elapsed_seconds": parse_seconds
                    + verify_seconds
                    + stats["filter_elapsed_seconds"],
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"captions": len(captions), "output": str(output)}))


if __name__ == "__main__":
    main()
