#!/usr/bin/env python3
"""Apply frozen VERA acceptance to base-only, VHR-only, and dual candidates."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_vera import apply_sentence_revisions, choose_revision


VARIANTS = {
    "base_revision": ("base",),
    "vhr_revision": ("vhr",),
    "vera": ("base", "vhr"),
}


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def atomic_jsonl_new(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", required=True)
    parser.add_argument("--source_extractions", required=True)
    parser.add_argument("--source_verifier", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--candidate_extractions", required=True)
    parser.add_argument("--candidate_verifier", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    captions = load_jsonl(args.captions)
    source_extractions = json.load(open(args.source_extractions))
    source_verifier = load_jsonl(args.source_verifier)
    tasks = json.load(open(args.tasks))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    revisions = load_jsonl(args.revisions)
    candidate_extractions = json.load(open(args.candidate_extractions))
    candidate_verifier = load_jsonl(args.candidate_verifier)
    expected_revision_ids = [row["revision_id"] for row in tasks]
    if [row["revision_id"] for row in revisions] != expected_revision_ids:
        raise ValueError("revisions do not match task order")
    if [row["revision_id"] for row in candidate_extractions] != expected_revision_ids:
        raise ValueError("candidate extractions do not match task order")
    if [int(row["question_id"]) for row in captions] != [
        int(row["image_id"]) for row in source_extractions
    ]:
        raise ValueError("source caption/extraction order mismatch")
    if args.limit is not None:
        selected_images = {int(row["image_id"]) for row in tasks}
        pairs = [
            (caption, extraction)
            for caption, extraction in zip(captions, source_extractions)
            if int(caption["question_id"]) in selected_images
        ]
        captions = [row[0] for row in pairs]
        source_extractions = [row[1] for row in pairs]

    source_score = {row["query_id"]: float(row["score"]) for row in source_verifier}
    expected_source_queries = {
        candidate["query_id"]
        for extraction in source_extractions
        for candidate in extraction["candidates"]
    }
    if not expected_source_queries.issubset(source_score):
        raise ValueError("source verifier is missing extraction queries")
    source_elapsed = {}
    source_peak = {}
    for row in source_verifier:
        image_id = int(row["image_id"])
        source_elapsed[image_id] = source_elapsed.get(image_id, 0.0) + float(
            row["elapsed_seconds"]
        )
        source_peak[image_id] = max(
            source_peak.get(image_id, 0.0), float(row["peak_allocated_gib"])
        )
    candidate_score = {
        row["query_id"]: float(row["score"]) for row in candidate_verifier
    }
    if len(candidate_score) != len(candidate_verifier):
        raise ValueError("duplicate candidate verifier query")
    expected_candidate_queries = {
        candidate["query_id"]
        for row in candidate_extractions
        for mode in ("base", "vhr")
        for candidate in row["modes"][mode]["candidates"]
    }
    if set(candidate_score) != expected_candidate_queries:
        raise ValueError("candidate verifier queries do not exactly match extraction")
    candidate_elapsed = {}
    candidate_peak = {}
    for row in candidate_verifier:
        key = (row["revision_id"], row["mode"])
        candidate_elapsed[key] = candidate_elapsed.get(key, 0.0) + float(
            row["elapsed_seconds"]
        )
        candidate_peak[key] = max(
            candidate_peak.get(key, 0.0), float(row["peak_allocated_gib"])
        )

    tasks_by_image = {}
    revision_by_id = {row["revision_id"]: row for row in revisions}
    extraction_by_id = {row["revision_id"]: row for row in candidate_extractions}
    for task in tasks:
        tasks_by_image.setdefault(int(task["image_id"]), []).append(task)

    outputs = {variant: [] for variant in VARIANTS}
    for caption, source_extraction in zip(captions, source_extractions):
        image_id = int(caption["question_id"])
        if hashlib.sha256(caption["text"].encode()).hexdigest() != source_extraction[
            "text_sha256"
        ]:
            raise ValueError(f"source caption changed for {image_id}")
        image_tasks = tasks_by_image.get(image_id, [])
        for variant, allowed_modes in VARIANTS.items():
            replacements = {}
            decisions = []
            additional_elapsed = float(source_extraction["extraction_elapsed_seconds"])
            additional_elapsed += source_elapsed.get(image_id, 0.0)
            peak = max(
                float(caption["peak_allocated_gib"]), source_peak.get(image_id, 0.0)
            )
            for task in image_tasks:
                revision = revision_by_id[task["revision_id"]]
                candidate_extraction = extraction_by_id[task["revision_id"]]
                mode_candidates = {}
                for mode in ("base", "vhr"):
                    revision_candidate = revision["candidates"][mode]
                    extraction = candidate_extraction["modes"][mode]
                    if hashlib.sha256(revision_candidate["text"].encode()).hexdigest() != extraction[
                        "text_sha256"
                    ]:
                        raise ValueError(
                            f"candidate text changed for {task['revision_id']} {mode}"
                        )
                    scores = {
                        candidate["query_id"]: candidate_score[candidate["query_id"]]
                        for candidate in extraction["candidates"]
                    }
                    mode_candidates[mode] = {
                        "text": revision_candidate["text"],
                        "extraction": extraction,
                        "scores": scores,
                        "trigger_roots": task["trigger_roots"],
                        "protected_roots": task["protected_roots"],
                    }
                    if mode in allowed_modes:
                        additional_elapsed += float(revision_candidate["elapsed_seconds"])
                        additional_elapsed += float(
                            extraction["extraction_elapsed_seconds"]
                        )
                        additional_elapsed += candidate_elapsed.get(
                            (task["revision_id"], mode), 0.0
                        )
                        peak = max(
                            peak,
                            float(revision_candidate["peak_allocated_gib"]),
                            candidate_peak.get((task["revision_id"], mode), 0.0),
                        )
                started = time.perf_counter()
                selected, audits = choose_revision(
                    task["source_sentence"], mode_candidates, allowed_modes
                )
                selection_elapsed = time.perf_counter() - started
                additional_elapsed += selection_elapsed
                if selected is not None:
                    replacements[int(task["sentence_index"])] = selected["text"]
                decisions.append(
                    {
                        "revision_id": task["revision_id"],
                        "sentence_index": int(task["sentence_index"]),
                        "selected_mode": None if selected is None else selected["mode"],
                        "fallback": selected is None,
                        "selection_elapsed_seconds": selection_elapsed,
                        "audits": {mode: asdict(audit) for mode, audit in audits.items()},
                    }
                )
            text = apply_sentence_revisions(
                caption["text"], source_extraction, replacements
            )
            selected_modes = [
                row["selected_mode"] for row in decisions if row["selected_mode"] is not None
            ]
            outputs[variant].append(
                {
                    **caption,
                    "text": text,
                    "method": variant,
                    "source_method": caption.get("method"),
                    "source_text_sha256": source_extraction["text_sha256"],
                    "elapsed_seconds": float(caption["elapsed_seconds"])
                    + additional_elapsed,
                    "peak_allocated_gib": peak,
                    "vera_stats": {
                        "triggered_sentences": len(decisions),
                        "accepted_revisions": len(selected_modes),
                        "fallback_sentences": sum(row["fallback"] for row in decisions),
                        "selected_base": selected_modes.count("base"),
                        "selected_vhr": selected_modes.count("vhr"),
                        "additional_elapsed_seconds": additional_elapsed,
                        "decisions": decisions,
                    },
                }
            )

    output_dir = Path(args.output_dir)
    for variant, rows in outputs.items():
        atomic_jsonl_new(output_dir / f"{variant}.json", rows)
    print(
        json.dumps(
            {
                variant: {
                    "captions": len(rows),
                    "triggered_sentences": sum(
                        row["vera_stats"]["triggered_sentences"] for row in rows
                    ),
                    "accepted_revisions": sum(
                        row["vera_stats"]["accepted_revisions"] for row in rows
                    ),
                    "selected_base": sum(row["vera_stats"]["selected_base"] for row in rows),
                    "selected_vhr": sum(row["vera_stats"]["selected_vhr"] for row in rows),
                }
                for variant, rows in outputs.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
