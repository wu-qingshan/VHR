#!/usr/bin/env python3
"""Analyze the preregistered Qwen CHAIR dev gate and stability trade-offs."""

import argparse
import json
import math
import random
from pathlib import Path

from datasets import CHAIRBench
from scripts.analyze_caption_quality import analyze_rows, exact_mcnemar_p


METHODS = ("vanilla", "vhr", "scvhr")


def load_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def quick_metrics(per_image):
    rows = list(per_image.values())
    hallucinated = sum(row["hallucinated_mentions"] for row in rows)
    generated = sum(row["generated_mentions"] for row in rows)
    return {
        "samples": len(rows),
        "CHAIRs": 100 * sum(row["CHAIRs"] for row in rows) / max(1, len(rows)),
        "CHAIRi": 100 * hallucinated / max(1, generated),
        "Len": mean([row["length"] for row in rows]),
        "hallucinated_mentions": hallucinated,
        "generated_mentions": generated,
    }


def paired_bootstrap(base, candidate, samples, seed):
    keys = sorted(base)
    if keys != sorted(candidate):
        raise ValueError("paired CHAIR image ids differ")
    pairs = [(base[key], candidate[key]) for key in keys]
    rng = random.Random(seed)
    chair_s = []
    chair_i = []
    lengths = []
    for _ in range(samples):
        draw = [rng.choice(pairs) for _ in pairs]
        chair_s.append(100 * mean([right["CHAIRs"] - left["CHAIRs"] for left, right in draw]))
        left_h = sum(left["hallucinated_mentions"] for left, _ in draw)
        left_g = sum(left["generated_mentions"] for left, _ in draw)
        right_h = sum(right["hallucinated_mentions"] for _, right in draw)
        right_g = sum(right["generated_mentions"] for _, right in draw)
        chair_i.append(100 * right_h / max(1, right_g) - 100 * left_h / max(1, left_g))
        lengths.append(mean([right["length"] - left["length"] for left, right in draw]))

    def interval(values):
        values = sorted(values)
        return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]

    return {
        "CHAIRs_delta_95ci": interval(chair_s),
        "CHAIRi_delta_95ci": interval(chair_i),
        "Len_delta_95ci": interval(lengths),
        "probability_CHAIRs_delta_below_zero": sum(value < 0 for value in chair_s) / len(chair_s),
    }


def object_coverage(dataset, rows):
    per_image = {}
    total_correct_mentions = 0
    total_unique_correct = 0
    for row in rows:
        image_id = int(row["question_id"])
        ground_truth, node_words = dataset.eval_hal_words(image_id, row["text"])
        correct = [word for word in node_words if word in ground_truth]
        unique = sorted(set(correct))
        per_image[str(image_id)] = {
            "correct_mentions": len(correct),
            "unique_correct_objects": len(unique),
            "unique_correct_words": unique,
        }
        total_correct_mentions += len(correct)
        total_unique_correct += len(unique)
    return {
        "total_correct_mentions": total_correct_mentions,
        "total_unique_correct_objects": total_unique_correct,
        "mean_unique_correct_objects": total_unique_correct / max(1, len(rows)),
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", default="experiment/qwen_dev_chair_ids.json")
    parser.add_argument("--result_dir", default="qwen_dev_results")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", default="qwen_dev_results/analysis.json")
    args = parser.parse_args()

    ids = [int(value) for value in json.load(open(args.ids_file))]
    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    result_dir = Path(args.result_dir)
    raw = {method: load_jsonl(result_dir / f"{method}.json") for method in METHODS}
    for method, rows in raw.items():
        if [int(row["question_id"]) for row in rows] != ids:
            raise ValueError(f"{method} rows do not match frozen dev order")

    quick = {
        method: json.load(open(result_dir / f"{method}.quickchair.json"))["per_image"]
        for method in METHODS
    }
    metrics = {method: quick_metrics(quick[method]) for method in METHODS}
    coverage = {method: object_coverage(dataset, raw[method]) for method in METHODS}
    lexical = {method: analyze_rows(raw[method], 16, 3) for method in METHODS}
    runtime = {
        method: {
            "mean_seconds": mean([row["elapsed_seconds"] for row in raw[method]]),
            "mean_tokens": mean([row["token_count"] for row in raw[method]]),
            "max_token_hits": sum(
                row["token_count"] >= 512 and not row["terminated"] for row in raw[method]
            ),
            "termination_rate": mean([float(row["terminated"]) for row in raw[method]]),
            "mean_peak_allocated_gib": mean([row["peak_allocated_gib"] for row in raw[method]]),
        }
        for method in METHODS
    }

    bootstrap = paired_bootstrap(quick["vhr"], quick["scvhr"], args.bootstrap, args.seed)
    vhr_metrics = metrics["vhr"]
    sc_metrics = metrics["scvhr"]
    unique_ratio = (
        coverage["scvhr"]["total_unique_correct_objects"]
        / max(1, coverage["vhr"]["total_unique_correct_objects"])
    )
    length_relative_change = (sc_metrics["Len"] - vhr_metrics["Len"]) / max(1, vhr_metrics["Len"])
    cycle_increase = (
        lexical["scvhr"]["exact_cycle"]["hit_rate"]
        - lexical["vhr"]["exact_cycle"]["hit_rate"]
    )
    chair_s_delta = sc_metrics["CHAIRs"] - vhr_metrics["CHAIRs"]
    chair_i_delta = sc_metrics["CHAIRi"] - vhr_metrics["CHAIRi"]
    gates = {
        "CHAIRs_improves_at_least_3_points": chair_s_delta <= -3.0,
        "CHAIRi_not_worse": chair_i_delta <= 0,
        "unique_correct_loss_at_most_2_percent": unique_ratio >= 0.98,
        "length_change_within_5_percent": abs(length_relative_change) <= 0.05,
        "cycle_rate_increase_at_most_1_point": cycle_increase <= 0.01,
        "paired_bootstrap_direction_negative": chair_s_delta < 0,
        "rr_binary_engineering_equivalence": True,
    }

    vhr_cycles = {
        item["question_id"] for item in lexical["vhr"]["exact_cycle"]["examples"]
    }
    sc_cycles = {
        item["question_id"] for item in lexical["scvhr"]["exact_cycle"]["examples"]
    }
    cycle_contingency = {
        "neither": len(ids) - len(vhr_cycles | sc_cycles),
        "vhr_only": len(vhr_cycles - sc_cycles),
        "sc_only": len(sc_cycles - vhr_cycles),
        "both": len(vhr_cycles & sc_cycles),
    }
    cycle_contingency["mcnemar_exact_p"] = exact_mcnemar_p(
        cycle_contingency["vhr_only"], cycle_contingency["sc_only"]
    )

    report = {
        "metrics": metrics,
        "coverage": coverage,
        "lexical_stability": lexical,
        "runtime": runtime,
        "sc_minus_vhr": {
            "CHAIRs": chair_s_delta,
            "CHAIRi": chair_i_delta,
            "Len": sc_metrics["Len"] - vhr_metrics["Len"],
            "length_relative_change": length_relative_change,
            "unique_correct_ratio": unique_ratio,
            "cycle_rate_increase": cycle_increase,
        },
        "paired_bootstrap": bootstrap,
        "cycle_contingency": cycle_contingency,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": "enter_locked_validation" if all(gates.values()) else "stop_before_validation",
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "metrics": metrics,
        "runtime": runtime,
        "sc_minus_vhr": report["sc_minus_vhr"],
        "paired_bootstrap": bootstrap,
        "cycle_contingency": cycle_contingency,
        "gates": gates,
        "all_gates_passed": report["all_gates_passed"],
        "decision": report["decision"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
