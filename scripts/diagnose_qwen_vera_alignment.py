#!/usr/bin/env python3
"""Explain failed VERA dev gates by CHAIR/trigger alignment without tuning."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", default="experiment/qwen_vera_dev_chair_ids.json")
    parser.add_argument("--tasks", default="qwen_vera_dev_results/tasks.json")
    parser.add_argument("--revisions", default="qwen_vera_dev_results/revisions.json")
    parser.add_argument("--vera", default="qwen_vera_dev_results/vera_outputs/vera.json")
    parser.add_argument("--output", default="qwen_vera_dev_results/alignment_diagnostic.json")
    args = parser.parse_args()

    ids = [int(value) for value in json.load(open(args.ids_file))]
    tasks = json.load(open(args.tasks))
    revisions = {row["revision_id"]: row for row in map(json.loads, open(args.revisions))}
    outputs = [json.loads(line) for line in open(args.vera) if line.strip()]
    decisions = {
        decision["revision_id"]: decision
        for row in outputs
        for decision in row["vera_stats"]["decisions"]
    }
    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _ = dataset.evaluator

    stats = {
        "tasks": len(tasks),
        "source_chair_hallucination_tasks": 0,
        "accepted": 0,
        "accepted_source_hallucination": 0,
        "accepted_fixes": 0,
        "accepted_introductions": 0,
        "fallback_source_hallucination": 0,
        "trigger_root_matches_chair_hallucination": 0,
    }
    selected_modes = {"base": 0, "vhr": 0}
    examples = []
    for task in tasks:
        ground_truth, source_words = dataset.eval_hal_words(
            task["image_id"], task["source_sentence"]
        )
        source_hallucinated = [word for word in source_words if word not in ground_truth]
        if source_hallucinated:
            stats["source_chair_hallucination_tasks"] += 1
        if set(task["trigger_roots"]) & set(source_hallucinated):
            stats["trigger_root_matches_chair_hallucination"] += 1
        decision = decisions[task["revision_id"]]
        mode = decision["selected_mode"]
        if mode is None:
            if source_hallucinated:
                stats["fallback_source_hallucination"] += 1
            continue
        stats["accepted"] += 1
        selected_modes[mode] += 1
        candidate = revisions[task["revision_id"]]["candidates"][mode]["text"]
        _, candidate_words = dataset.eval_hal_words(task["image_id"], candidate)
        candidate_hallucinated = [
            word for word in candidate_words if word not in ground_truth
        ]
        if source_hallucinated:
            stats["accepted_source_hallucination"] += 1
        if source_hallucinated and not candidate_hallucinated:
            stats["accepted_fixes"] += 1
        if not source_hallucinated and candidate_hallucinated:
            stats["accepted_introductions"] += 1
        if (source_hallucinated or candidate_hallucinated) and len(examples) < 30:
            examples.append(
                {
                    "revision_id": task["revision_id"],
                    "selected_mode": mode,
                    "trigger_roots": task["trigger_roots"],
                    "source_hallucinated": source_hallucinated,
                    "candidate_hallucinated": candidate_hallucinated,
                }
            )
    report = {
        "posthoc_explanation_only": True,
        "must_not_tune_current_dev": True,
        "stats": stats,
        "rates": {
            "chair_hallucination_task_rate": stats["source_chair_hallucination_tasks"]
            / max(1, stats["tasks"]),
            "trigger_root_alignment_rate": stats[
                "trigger_root_matches_chair_hallucination"
            ]
            / max(1, stats["tasks"]),
            "accepted_chair_hallucination_rate": stats[
                "accepted_source_hallucination"
            ]
            / max(1, stats["accepted"]),
        },
        "selected_modes": selected_modes,
        "examples": examples,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
