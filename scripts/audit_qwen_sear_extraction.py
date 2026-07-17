#!/usr/bin/env python3
"""Audit preregistered generic noun-phrase extraction on frozen Qwen dev."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench
from qwen_sear import extract_caption, mention_is_captured


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--ids_file", default="experiment/qwen_dev_chair_ids.json")
    parser.add_argument("--captions", default="qwen_dev_results/vanilla.json")
    parser.add_argument(
        "--output", default="qwen_sear_dev_results/extraction_audit.json"
    )
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()

    import spacy

    nlp = spacy.load(args.spacy_model)
    model_version = str(nlp.meta.get("version"))
    if model_version != "3.8.0":
        raise RuntimeError(
            f"expected en_core_web_sm 3.8.0, found {model_version}"
        )

    ids = [int(value) for value in json.load(open(args.ids_file))]
    rows = load_jsonl(args.captions)
    row_ids = [int(row["question_id"]) for row in rows]
    if row_ids != ids:
        raise ValueError("captions do not match frozen Qwen dev IDs in exact order")

    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        evaluator = dataset.evaluator

    total_mentions = 0
    captured_mentions = 0
    candidate_counts = []
    pair_status = {}
    misses = []
    per_image = []
    for row in rows:
        image_id = int(row["question_id"])
        extraction = extract_caption(row["text"], nlp)
        candidates = extraction["candidates"]
        candidate_counts.append(len(candidates))

        words, node_words, idxs, _ = evaluator.caption_to_words(row["text"])
        image_total = 0
        image_captured = 0
        seen_mentions = set()
        for word, node_word, token_index in zip(words, node_words, idxs):
            # "Unique mention" is lexical within an image, not token-position
            # unique. This prevents repeated generation cycles from dominating
            # the extractor gate while retaining distinct CHAIR synonyms.
            mention_key = (str(word), str(node_word))
            if mention_key in seen_mentions:
                continue
            seen_mentions.add(mention_key)
            captured = mention_is_captured(word, candidates)
            total_mentions += 1
            captured_mentions += int(captured)
            image_total += 1
            image_captured += int(captured)
            pair_key = (image_id, str(node_word))
            pair_status[pair_key] = pair_status.get(pair_key, False) or captured
            if not captured and len(misses) < 100:
                misses.append(
                    {
                        "image_id": image_id,
                        "mention": word,
                        "canonical_object": node_word,
                        "token_index": int(token_index),
                    }
                )
        per_image.append(
            {
                "image_id": image_id,
                "candidate_count": len(candidates),
                "mention_count": image_total,
                "captured_mentions": image_captured,
            }
        )

    mention_recall = captured_mentions / max(1, total_mentions)
    captured_pairs = sum(pair_status.values())
    pair_recall = captured_pairs / max(1, len(pair_status))
    mean_candidates = sum(candidate_counts) / max(1, len(candidate_counts))
    sorted_counts = sorted(candidate_counts)
    p95_index = max(0, int(0.95 * len(sorted_counts)) - 1)
    metrics = {
        "images": len(rows),
        "chair_unique_mentions": total_mentions,
        "captured_unique_mentions": captured_mentions,
        "unique_mention_recall": mention_recall,
        "chair_unique_image_object_pairs": len(pair_status),
        "captured_unique_image_object_pairs": captured_pairs,
        "unique_image_object_recall": pair_recall,
        "total_candidates": sum(candidate_counts),
        "mean_candidates_per_image": mean_candidates,
        "median_candidates_per_image": sorted_counts[len(sorted_counts) // 2],
        "p95_candidates_per_image": sorted_counts[p95_index],
        "max_candidates_per_image": max(candidate_counts, default=0),
    }
    gates = {
        "unique_mention_recall_at_least_0_90": mention_recall >= 0.90,
        "mean_candidates_at_most_20": mean_candidates <= 20.0,
    }
    report = {
        "protocol": {
            "captions": args.captions,
            "captions_sha256": sha256(args.captions),
            "ids_file": args.ids_file,
            "ids_sha256": sha256(args.ids_file),
            "spacy_version": spacy.__version__,
            "spacy_model": args.spacy_model,
            "spacy_model_version": model_version,
            "chair_vocabulary_used_only_for_audit": True,
        },
        "metrics": metrics,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": (
            "implement_sear_filter"
            if all(gates.values())
            else "stop_before_sear_effect"
        ),
        "miss_examples_not_for_rule_tuning": misses,
        "per_image": per_image,
    }
    atomic_json_new(args.output, report)
    print(
        json.dumps(
            {
                "metrics": metrics,
                "gates": gates,
                "all_gates_passed": report["all_gates_passed"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
