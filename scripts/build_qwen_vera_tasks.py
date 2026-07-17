#!/usr/bin/env python3
"""Build evidence-triggered VERA sentence-revision tasks without CHAIR labels."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_vera import format_revision_prompt, valid_visual_candidate


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


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
    parser.add_argument("--captions", required=True)
    parser.add_argument("--source_extractions", required=True)
    parser.add_argument("--source_verifier", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    captions = load_jsonl(args.captions)
    extractions = json.load(open(args.source_extractions))
    verifier = load_jsonl(args.source_verifier)
    if [int(row["question_id"]) for row in captions] != [
        int(row["image_id"]) for row in extractions
    ]:
        raise ValueError("caption/extraction image order mismatch")
    scores = {}
    for row in verifier:
        query_id = row["query_id"]
        if query_id in scores:
            raise ValueError(f"duplicate source verifier query {query_id}")
        scores[query_id] = float(row["score"])
    expected = {
        candidate["query_id"]
        for extraction in extractions
        for candidate in extraction["candidates"]
    }
    if set(scores) != expected:
        raise ValueError("source verifier queries do not exactly match source extraction")

    tasks = []
    for caption, extraction in zip(captions, extractions):
        if hashlib.sha256(caption["text"].encode()).hexdigest() != extraction["text_sha256"]:
            raise ValueError(f"source caption changed for {caption['question_id']}")
        sentence_candidates = {sentence["index"]: [] for sentence in extraction["sentences"]}
        for candidate in extraction["candidates"]:
            if not valid_visual_candidate(candidate):
                continue
            score = scores[candidate["query_id"]]
            for occurrence in candidate["occurrences"]:
                sentence_candidates[int(occurrence["sentence_index"])].append(
                    {
                        "query_id": candidate["query_id"],
                        "root_lemma": candidate["root_lemma"],
                        "phrase": occurrence["phrase"],
                        "score": score,
                    }
                )
        for sentence in extraction["sentences"]:
            candidates = sentence_candidates[sentence["index"]]
            triggers = [row for row in candidates if row["score"] <= 0.1]
            if not triggers:
                continue
            protected = sorted(
                {row["root_lemma"] for row in candidates if row["score"] >= 0.5}
            )
            trigger_roots = sorted({row["root_lemma"] for row in triggers})
            trigger_phrases = []
            seen_phrases = set()
            for row in triggers:
                if row["phrase"] not in seen_phrases:
                    seen_phrases.add(row["phrase"])
                    trigger_phrases.append(row["phrase"])
            image_id = int(caption["question_id"])
            tasks.append(
                {
                    "revision_id": f"{image_id}:{int(sentence['index']):03d}",
                    "image_id": image_id,
                    "sentence_index": int(sentence["index"]),
                    "source_sentence": sentence["text"],
                    "source_sentence_sha256": hashlib.sha256(
                        sentence["text"].encode()
                    ).hexdigest(),
                    "trigger_phrases": trigger_phrases,
                    "trigger_roots": trigger_roots,
                    "protected_roots": protected,
                    "source_candidate_query_ids": [row["query_id"] for row in candidates],
                    "prompt": format_revision_prompt(sentence["text"], trigger_phrases),
                }
            )
    atomic_json_new(args.output, tasks)
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "images": len({row["image_id"] for row in tasks}),
                "mean_protected_roots": sum(len(row["protected_roots"]) for row in tasks)
                / max(1, len(tasks)),
                "chair_vocabulary_used": False,
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
