#!/usr/bin/env python3
"""Evaluate all preregistered SEAR dev gates and deletion trade-offs."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench
from scripts.analyze_caption_quality import analyze_rows


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def quick_metrics(per_image):
    rows = list(per_image.values())
    hallucinated = sum(row["hallucinated_mentions"] for row in rows)
    generated = sum(row["generated_mentions"] for row in rows)
    return {
        "CHAIRs": 100 * mean([row["CHAIRs"] for row in rows]),
        "CHAIRi": 100 * hallucinated / max(1, generated),
        "Len": mean([row["length"] for row in rows]),
        "hallucinated_mentions": hallucinated,
        "generated_mentions": generated,
    }


def describe_objects(dataset, row):
    ground_truth, node_words = dataset.eval_hal_words(row["question_id"], row["text"])
    correct = [word for word in node_words if word in ground_truth]
    hallucinated = [word for word in node_words if word not in ground_truth]
    return {
        "correct": correct,
        "hallucinated": hallucinated,
        "unique_correct": set(correct),
        "unique_hallucinated": set(hallucinated),
    }


def bootstrap_delta(base, candidate, draws, seed):
    keys = sorted(base, key=int)
    pairs = [(base[key], candidate[key]) for key in keys]
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [rng.choice(pairs) for _ in pairs]
        values.append(100 * mean([right["CHAIRs"] - left["CHAIRs"] for left, right in sample]))
    values.sort()
    return {
        "point": 100 * mean([right["CHAIRs"] - left["CHAIRs"] for left, right in pairs]),
        "95ci": [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]],
        "probability_below_zero": sum(value < 0 for value in values) / len(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", default="experiment/qwen_dev_chair_ids.json")
    parser.add_argument("--baseline_dir", default="qwen_dev_results")
    parser.add_argument("--sear_dir", default="qwen_sear_dev_results")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--output", default="qwen_sear_dev_results/analysis.json")
    args = parser.parse_args()

    ids = [int(value) for value in json.load(open(args.ids_file))]
    baseline_dir = Path(args.baseline_dir)
    sear_dir = Path(args.sear_dir)
    rows = {
        "vanilla": load_jsonl(baseline_dir / "vanilla.json"),
        "scvhr": load_jsonl(baseline_dir / "scvhr.json"),
        "sear": load_jsonl(sear_dir / "sear.json"),
    }
    for method, method_rows in rows.items():
        if [int(row["question_id"]) for row in method_rows] != ids:
            raise ValueError(f"{method} rows do not match frozen Qwen dev IDs")

    quick = {
        "vanilla": json.load(open(baseline_dir / "vanilla.quickchair.json"))["per_image"],
        "sear": json.load(open(sear_dir / "sear.quickchair.json"))["per_image"],
    }
    metrics = {method: quick_metrics(value) for method, value in quick.items()}

    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _ = dataset.evaluator
    object_rows = {
        method: [describe_objects(dataset, row) for row in method_rows]
        for method, method_rows in (("vanilla", rows["vanilla"]), ("sear", rows["sear"]))
    }
    coverage = {}
    for method, values in object_rows.items():
        coverage[method] = {
            "correct_mentions": sum(len(row["correct"]) for row in values),
            "hallucinated_mentions": sum(len(row["hallucinated"]) for row in values),
            "unique_correct_objects": sum(len(row["unique_correct"]) for row in values),
            "unique_hallucinated_objects": sum(
                len(row["unique_hallucinated"]) for row in values
            ),
        }

    stability = {
        method: analyze_rows(method_rows, 16, 3)
        for method, method_rows in (("vanilla", rows["vanilla"]), ("sear", rows["sear"]))
    }
    source_max_hits = {
        method: sum(
            not row["terminated"] and int(row["token_count"]) >= 512
            for row in method_rows
        )
        for method, method_rows in (("vanilla", rows["vanilla"]), ("sear", rows["sear"]))
    }
    runtime = {
        "vanilla_total_seconds_per_image": mean(
            [float(row["elapsed_seconds"]) for row in rows["vanilla"]]
        ),
        "scvhr_total_seconds_per_image": mean(
            [float(row["elapsed_seconds"]) for row in rows["scvhr"]]
        ),
        "sear_total_seconds_per_image": mean(
            [float(row["elapsed_seconds"]) for row in rows["sear"]]
        ),
        "sear_only_seconds_per_image": mean(
            [float(row["sear_stats"]["sear_only_elapsed_seconds"]) for row in rows["sear"]]
        ),
        "sear_verifier_seconds_per_image": mean(
            [float(row["sear_stats"]["verifier_elapsed_seconds"]) for row in rows["sear"]]
        ),
        "sear_extraction_seconds_per_image": mean(
            [float(row["sear_stats"]["extraction_elapsed_seconds"]) for row in rows["sear"]]
        ),
        "sear_peak_allocated_gib": max(float(row["peak_allocated_gib"]) for row in rows["sear"]),
    }

    deleted_object_mentions = {"correct": 0, "hallucinated": 0}
    deleted_unique = {"correct": 0, "hallucinated": 0}
    triggered = []
    fallback = []
    source_sentences = 0
    retained_sentences = 0
    deleted_sentences = 0
    remaining_failure_examples = []
    corrected_examples = []
    for vanilla, sear, vanilla_objects, sear_objects in zip(
        rows["vanilla"], rows["sear"], object_rows["vanilla"], object_rows["sear"]
    ):
        stats = sear["sear_stats"]
        source_sentences += int(stats["source_sentence_count"])
        retained_sentences += int(stats["retained_sentence_count"])
        deleted_sentences += len(stats["deleted_sentence_indices"])
        if stats["triggered"]:
            triggered.append(int(sear["question_id"]))
        if stats["fallback_used"]:
            fallback.append(int(sear["question_id"]))
        removed_correct = len(vanilla_objects["correct"]) - len(sear_objects["correct"])
        removed_hallucinated = len(vanilla_objects["hallucinated"]) - len(
            sear_objects["hallucinated"]
        )
        deleted_object_mentions["correct"] += removed_correct
        deleted_object_mentions["hallucinated"] += removed_hallucinated
        deleted_unique["correct"] += len(
            vanilla_objects["unique_correct"] - sear_objects["unique_correct"]
        )
        deleted_unique["hallucinated"] += len(
            vanilla_objects["unique_hallucinated"] - sear_objects["unique_hallucinated"]
        )
        vanilla_has = bool(vanilla_objects["hallucinated"])
        sear_has = bool(sear_objects["hallucinated"])
        if vanilla_has and not sear_has and len(corrected_examples) < 20:
            corrected_examples.append(
                {
                    "image_id": int(sear["question_id"]),
                    "removed_hallucinated_objects": sorted(
                        vanilla_objects["unique_hallucinated"]
                        - sear_objects["unique_hallucinated"]
                    ),
                    "removed_unique_correct_objects": sorted(
                        vanilla_objects["unique_correct"] - sear_objects["unique_correct"]
                    ),
                }
            )
        if sear_has and len(remaining_failure_examples) < 20:
            remaining_failure_examples.append(
                {
                    "image_id": int(sear["question_id"]),
                    "remaining_hallucinated_objects": sorted(
                        sear_objects["unique_hallucinated"]
                    ),
                }
            )

    deletion = {
        "triggered_images": len(triggered),
        "triggered_image_rate": len(triggered) / len(ids),
        "fallback_images": len(fallback),
        "source_sentences": source_sentences,
        "retained_sentences": retained_sentences,
        "deleted_sentences": deleted_sentences,
        "deleted_sentence_rate": deleted_sentences / max(1, source_sentences),
        "removed_object_mentions": deleted_object_mentions,
        "removed_unique_objects": deleted_unique,
        "hallucinated_to_correct_removed_mention_ratio": deleted_object_mentions[
            "hallucinated"
        ]
        / max(1, deleted_object_mentions["correct"]),
    }
    paired = bootstrap_delta(quick["vanilla"], quick["sear"], args.bootstrap, 20260717)
    vanilla = metrics["vanilla"]
    sear = metrics["sear"]
    unique_correct_ratio = coverage["sear"]["unique_correct_objects"] / max(
        1, coverage["vanilla"]["unique_correct_objects"]
    )
    gates = {
        "CHAIRs_at_most_20_67": sear["CHAIRs"] <= 20.67,
        "CHAIRi_at_most_vanilla": sear["CHAIRi"] <= vanilla["CHAIRi"],
        "unique_correct_at_least_95pct_vanilla": unique_correct_ratio >= 0.95,
        "length_between_0_75_and_1_05_vanilla": 0.75 * vanilla["Len"]
        <= sear["Len"]
        <= 1.05 * vanilla["Len"],
        "cycle_not_above_vanilla": stability["sear"]["exact_cycle"]["hits"]
        <= stability["vanilla"]["exact_cycle"]["hits"],
        "source_max_hit_not_above_vanilla": source_max_hits["sear"]
        <= source_max_hits["vanilla"],
        "sear_only_latency_below_sc": runtime["sear_only_seconds_per_image"]
        < runtime["scvhr_total_seconds_per_image"],
        "paired_CHAIRs_direction_below_zero": paired["point"] < 0,
    }
    report = {
        "metrics": metrics,
        "coverage": coverage,
        "unique_correct_retention": unique_correct_ratio,
        "stability": stability,
        "source_max_hits": source_max_hits,
        "runtime": runtime,
        "deletion": deletion,
        "paired_sear_minus_vanilla": paired,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": "enter_locked_validation" if all(gates.values()) else "stop_before_validation",
        "corrected_examples": corrected_examples,
        "remaining_failure_examples": remaining_failure_examples,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {
        key: report[key]
        for key in (
            "metrics",
            "coverage",
            "unique_correct_retention",
            "source_max_hits",
            "runtime",
            "deletion",
            "paired_sear_minus_vanilla",
            "gates",
            "all_gates_passed",
            "decision",
        )
    }
    summary["cycle_hits"] = {
        method: values["exact_cycle"]["hits"] for method, values in stability.items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
