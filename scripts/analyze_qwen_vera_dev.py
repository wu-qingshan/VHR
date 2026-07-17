#!/usr/bin/env python3
"""Evaluate frozen VERA-VHR dev outputs against all preregistered gates."""

import argparse
from collections import Counter
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
    values = list(per_image.values())
    hallucinated = sum(row["hallucinated_mentions"] for row in values)
    generated = sum(row["generated_mentions"] for row in values)
    return {
        "CHAIRs": 100 * mean([row["CHAIRs"] for row in values]),
        "CHAIRi": 100 * hallucinated / max(1, generated),
        "Len": mean([row["length"] for row in values]),
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


def aggregate_coverage(values):
    return {
        "correct_mentions": sum(len(row["correct"]) for row in values),
        "hallucinated_mentions": sum(len(row["hallucinated"]) for row in values),
        "unique_correct_objects": sum(len(row["unique_correct"]) for row in values),
        "unique_hallucinated_objects": sum(len(row["unique_hallucinated"]) for row in values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", default="experiment/qwen_vera_dev_chair_ids.json")
    parser.add_argument("--baseline_dir", default="qwen_vera_dev_results")
    parser.add_argument(
        "--revision_dir", default="qwen_vera_dev_results/vera_outputs"
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--output", default="qwen_vera_dev_results/analysis.json")
    args = parser.parse_args()

    ids = [int(value) for value in json.load(open(args.ids_file))]
    baseline_dir = Path(args.baseline_dir)
    revision_dir = Path(args.revision_dir)
    method_paths = {
        "vanilla": baseline_dir / "vanilla.json",
        "vhr": baseline_dir / "vhr.json",
        "scvhr": baseline_dir / "scvhr.json",
        "base_revision": revision_dir / "base_revision.json",
        "vhr_revision": revision_dir / "vhr_revision.json",
        "vera": revision_dir / "vera.json",
    }
    sear_path = revision_dir / "sear_sentence.json"
    if sear_path.is_file():
        method_paths["sear_sentence"] = sear_path
    rows = {method: load_jsonl(path) for method, path in method_paths.items()}
    for method, values in rows.items():
        if [int(row["question_id"]) for row in values] != ids:
            raise ValueError(f"{method} rows do not match frozen VERA dev IDs")

    quick = {
        method: json.load(open(path.with_name(path.name.replace(".json", ".quickchair.json"))))[
            "per_image"
        ]
        for method, path in method_paths.items()
    }
    metrics = {method: quick_metrics(value) for method, value in quick.items()}
    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _ = dataset.evaluator
    object_rows = {
        method: [describe_objects(dataset, row) for row in values]
        for method, values in rows.items()
    }
    coverage = {
        method: aggregate_coverage(values) for method, values in object_rows.items()
    }
    stability = {method: analyze_rows(values, 16, 3) for method, values in rows.items()}
    runtime = {
        method: {
            "mean_seconds_per_image": mean(
                [float(row["elapsed_seconds"]) for row in values]
            ),
            "peak_allocated_gib": max(float(row["peak_allocated_gib"]) for row in values),
            "source_512_hits": sum(
                not row["terminated"] and int(row["token_count"]) >= 512
                for row in values
            ),
        }
        for method, values in rows.items()
    }

    source_objects = object_rows["vanilla"]
    tradeoff = {}
    for method in ("base_revision", "vhr_revision", "vera"):
        removed = {"correct_mentions": 0, "hallucinated_mentions": 0, "unique_correct": 0, "unique_hallucinated": 0}
        added = {"correct_mentions": 0, "hallucinated_mentions": 0, "unique_correct": 0, "unique_hallucinated": 0}
        for source, candidate in zip(source_objects, object_rows[method]):
            source_correct = Counter(source["correct"])
            candidate_correct = Counter(candidate["correct"])
            source_hallucinated = Counter(source["hallucinated"])
            candidate_hallucinated = Counter(candidate["hallucinated"])
            removed["correct_mentions"] += sum((source_correct - candidate_correct).values())
            removed["hallucinated_mentions"] += sum(
                (source_hallucinated - candidate_hallucinated).values()
            )
            added["correct_mentions"] += sum((candidate_correct - source_correct).values())
            added["hallucinated_mentions"] += sum(
                (candidate_hallucinated - source_hallucinated).values()
            )
            removed["unique_correct"] += len(source["unique_correct"] - candidate["unique_correct"])
            removed["unique_hallucinated"] += len(
                source["unique_hallucinated"] - candidate["unique_hallucinated"]
            )
            added["unique_correct"] += len(candidate["unique_correct"] - source["unique_correct"])
            added["unique_hallucinated"] += len(
                candidate["unique_hallucinated"] - source["unique_hallucinated"]
            )
        tradeoff[method] = {"removed": removed, "added": added}

    routing = {}
    for method in ("base_revision", "vhr_revision", "vera"):
        values = rows[method]
        triggered = sum(row["vera_stats"]["triggered_sentences"] for row in values)
        accepted = sum(row["vera_stats"]["accepted_revisions"] for row in values)
        routing[method] = {
            "triggered_sentences": triggered,
            "accepted_revisions": accepted,
            "acceptance_rate": accepted / max(1, triggered),
            "fallback_sentences": sum(row["vera_stats"]["fallback_sentences"] for row in values),
            "selected_base": sum(row["vera_stats"]["selected_base"] for row in values),
            "selected_vhr": sum(row["vera_stats"]["selected_vhr"] for row in values),
        }

    import spacy

    nlp = spacy.load("en_core_web_sm")
    grammar = {}
    source_sentence_counts = [len(list(nlp(row["text"]).sents)) for row in rows["vanilla"]]
    for method in ("base_revision", "vhr_revision", "vera"):
        candidate_counts = [len(list(nlp(row["text"]).sents)) for row in rows[method]]
        grammar[method] = {
            "mean_source_sentences": mean(source_sentence_counts),
            "mean_output_sentences": mean(candidate_counts),
            "captions_with_sentence_count_change": sum(
                left != right for left, right in zip(source_sentence_counts, candidate_counts)
            ),
            "all_accepted_candidates_single_sentence": all(
                decision["audits"][decision["selected_mode"]]["checks"][
                    "exactly_one_nonempty_sentence"
                ]
                for row in rows[method]
                for decision in row["vera_stats"]["decisions"]
                if decision["selected_mode"] is not None
            ),
        }

    paired = {
        method: bootstrap_delta(quick["vanilla"], quick[method], args.bootstrap, 2026071701)
        for method in ("base_revision", "vhr_revision", "vera")
    }
    vanilla = metrics["vanilla"]
    vera = metrics["vera"]
    unique_correct_retention = coverage["vera"]["unique_correct_objects"] / max(
        1, coverage["vanilla"]["unique_correct_objects"]
    )
    gates = {
        "CHAIRs_improves_at_least_3": vera["CHAIRs"] <= vanilla["CHAIRs"] - 3.0,
        "CHAIRi_not_above_vanilla": vera["CHAIRi"] <= vanilla["CHAIRi"],
        "unique_correct_at_least_98pct_vanilla": unique_correct_retention >= 0.98,
        "length_between_0_90_and_1_05_vanilla": 0.90 * vanilla["Len"]
        <= vera["Len"]
        <= 1.05 * vanilla["Len"],
        "cycle_not_above_vanilla": stability["vera"]["exact_cycle"]["hits"]
        <= stability["vanilla"]["exact_cycle"]["hits"],
        "source_512_hit_not_above_vanilla": runtime["vera"]["source_512_hits"]
        <= runtime["vanilla"]["source_512_hits"],
        "paired_direction_negative": paired["vera"]["point"] < 0,
        "paired_ci_upper_not_above_zero": paired["vera"]["95ci"][1] <= 0,
        "total_latency_below_sc": runtime["vera"]["mean_seconds_per_image"]
        < runtime["scvhr"]["mean_seconds_per_image"],
        "acceptance_rate_at_least_20pct": routing["vera"]["acceptance_rate"] >= 0.20,
    }
    report = {
        "metrics": metrics,
        "coverage": coverage,
        "unique_correct_retention": unique_correct_retention,
        "stability": stability,
        "runtime": runtime,
        "routing": routing,
        "object_tradeoff_vs_vanilla": tradeoff,
        "grammar": grammar,
        "paired_vs_vanilla": paired,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": "enter_locked_validation" if all(gates.values()) else "stop_before_validation",
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
            "runtime",
            "routing",
            "object_tradeoff_vs_vanilla",
            "grammar",
            "paired_vs_vanilla",
            "gates",
            "all_gates_passed",
            "decision",
        )
    }
    summary["cycle_hits"] = {
        method: value["exact_cycle"]["hits"] for method, value in stability.items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
