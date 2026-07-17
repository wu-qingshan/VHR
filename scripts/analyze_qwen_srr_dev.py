#!/usr/bin/env python3
"""Evaluate the preregistered SRR dev gate against frozen Qwen baselines."""

import argparse
import json
import random
from pathlib import Path

from datasets import CHAIRBench
from scripts.analyze_caption_quality import analyze_rows


def load_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def metrics(per_image):
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


def unique_correct(dataset, rows):
    total = 0
    for row in rows:
        ground_truth, node_words = dataset.eval_hal_words(row["question_id"], row["text"])
        total += len(set(word for word in node_words if word in ground_truth))
    return total


def bootstrap_delta(base, candidate, draws, seed):
    keys = sorted(base)
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
    parser.add_argument("--srr_dir", default="qwen_srr_dev_results")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--output", default="qwen_srr_dev_results/analysis.json")
    args = parser.parse_args()

    ids = [int(value) for value in json.load(open(args.ids_file))]
    baseline_dir = Path(args.baseline_dir)
    srr_dir = Path(args.srr_dir)
    rows = {
        "vanilla": load_jsonl(baseline_dir / "vanilla.json"),
        "vhr": load_jsonl(baseline_dir / "vhr.json"),
        "scvhr": load_jsonl(baseline_dir / "scvhr.json"),
        "srrscvhr": load_jsonl(srr_dir / "srrscvhr.json"),
    }
    for method, method_rows in rows.items():
        if [int(row["question_id"]) for row in method_rows] != ids:
            raise ValueError(f"{method} order differs from frozen IDs")

    quick = {
        "vanilla": json.load(open(baseline_dir / "vanilla.quickchair.json"))["per_image"],
        "vhr": json.load(open(baseline_dir / "vhr.quickchair.json"))["per_image"],
        "scvhr": json.load(open(baseline_dir / "scvhr.quickchair.json"))["per_image"],
        "srrscvhr": json.load(open(srr_dir / "srrscvhr.quickchair.json"))["per_image"],
    }
    score = {method: metrics(value) for method, value in quick.items()}
    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    correct = {method: unique_correct(dataset, value) for method, value in rows.items()}
    stability = {method: analyze_rows(value, 16, 3) for method, value in rows.items()}
    runtime = {
        method: {
            "mean_seconds": mean([row["elapsed_seconds"] for row in value]),
            "mean_tokens": mean([row["token_count"] for row in value]),
            "max_token_hits": sum(not row["terminated"] and row["token_count"] >= 512 for row in value),
        }
        for method, value in rows.items()
    }

    rollback_rows = [row for row in rows["srrscvhr"] if row["routing_stats"]["rollback_triggered"]]
    rollback = {
        "count": len(rollback_rows),
        "rate": len(rollback_rows) / len(ids),
        "mean_step": mean([row["routing_stats"]["rollback_step"] for row in rollback_rows]),
        "all_stop_vhr_forward": all(
            row["forward_counts"]["vhr"] < row["forward_counts"]["base"] for row in rollback_rows
        ),
        "mean_saved_vhr_forwards": mean(
            [row["forward_counts"]["base"] - row["forward_counts"]["vhr"] for row in rollback_rows]
        ),
        "safe_masked_tokens": sum(row["routing_stats"]["safe_masked_tokens"] for row in rollback_rows),
    }
    paired = bootstrap_delta(quick["vhr"], quick["srrscvhr"], args.bootstrap, 20260717)

    vanilla = score["vanilla"]
    vhr = score["vhr"]
    srr = score["srrscvhr"]
    vanilla_cycle_rate = stability["vanilla"]["exact_cycle"]["hit_rate"]
    srr_cycle_rate = stability["srrscvhr"]["exact_cycle"]["hit_rate"]
    gates = {
        "CHAIRs_at_most_24_67": srr["CHAIRs"] <= 24.67,
        "CHAIRs_improves_vhr_at_least_5": srr["CHAIRs"] <= vhr["CHAIRs"] - 5.0,
        "CHAIRi_at_most_6_96": srr["CHAIRi"] <= 6.96,
        "unique_correct_at_least_98pct_vanilla": correct["srrscvhr"] >= 0.98 * correct["vanilla"],
        "length_between_0_9_and_1_25_vanilla": 0.9 * vanilla["Len"] <= srr["Len"] <= 1.25 * vanilla["Len"],
        "cycle_at_most_vanilla_plus_2pp": srr_cycle_rate <= vanilla_cycle_rate + 0.02,
        "max_hit_at_most_vanilla_plus_5pp": runtime["srrscvhr"]["max_token_hits"] <= runtime["vanilla"]["max_token_hits"] + 15,
        "latency_below_sc": runtime["srrscvhr"]["mean_seconds"] < runtime["scvhr"]["mean_seconds"],
        "rollback_stops_vhr_forward": rollback["all_stop_vhr_forward"],
        "paired_direction_below_vhr": paired["point"] < 0,
    }
    report = {
        "metrics": score,
        "unique_correct_objects": correct,
        "stability": stability,
        "runtime": runtime,
        "rollback": rollback,
        "paired_srr_minus_vhr": paired,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": "enter_locked_validation" if all(gates.values()) else "stop_before_validation",
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {key: report[key] for key in (
        "metrics",
        "unique_correct_objects",
        "runtime",
        "rollback",
        "paired_srr_minus_vhr",
        "gates",
        "all_gates_passed",
        "decision",
    )}
    summary["cycle_hits"] = {
        method: value["exact_cycle"]["hits"] for method, value in stability.items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
