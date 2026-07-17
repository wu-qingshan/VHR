import argparse
import contextlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench
from scripts.analyze_caption_quality import caption_metrics


METHOD_FILES = {
    "vhr": "0-chair_0-vhr-care-dev-h14-512.json",
    "scvhr": "0-chair_0-scvhr-care-dev-h14-a05-b01-512.json",
    "cascvhr": "0-chair_0-cascvhr-care-dev-h14-a05-b01-512.json",
    "carescvhr": "0-chair_0-carescvhr-care-dev-h14-a05-b01-512.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CARE-SC dev gates without tuning.")
    parser.add_argument("--result_dir", default="care_dev_results/llava-1.5-7b-hf")
    parser.add_argument("--ids_file", default="experiment/care_dev_chair_ids.json")
    parser.add_argument("--output", default="care_dev_results/care_dev_analysis.json")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def load_rows(path, image_ids):
    with open(path) as source:
        rows = [json.loads(line) for line in source if line.strip()]
    by_id = {int(row["question_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate question IDs in {path}")
    if set(by_id) != set(image_ids):
        raise ValueError(f"Result IDs do not exactly match {len(image_ids)} IDs: {path}")
    return [by_id[image_id] for image_id in image_ids]


def describe(dataset, row):
    gt_objects, node_words = dataset.eval_hal_words(row["question_id"], row["text"])
    correct = [word for word in node_words if word in gt_objects]
    hallucinated = [word for word in node_words if word not in gt_objects]
    quality = caption_metrics(row, max_cycle_period=16, cycle_repeats=3)
    return {
        "CHAIRs": int(bool(hallucinated)),
        "hallucinated_mentions": len(hallucinated),
        "generated_mentions": len(node_words),
        "unique_correct_objects": len(set(correct)),
        "length": len(row["text"].split()),
        "distinct_1": quality["distinct_1"],
        "distinct_2": quality["distinct_2"],
        "repeated_2gram_ratio": quality["repeated_2gram_ratio"],
        "repeated_3gram_ratio": quality["repeated_3gram_ratio"],
        "repeated_4gram_ratio": quality["repeated_4gram_ratio"],
        "exact_cycle": int(quality["exact_cycle"]),
    }


def mean(values):
    return sum(values) / len(values)


def aggregate(rows):
    hallucinated = sum(row["hallucinated_mentions"] for row in rows)
    generated = sum(row["generated_mentions"] for row in rows)
    return {
        "samples": len(rows),
        "CHAIRs": 100 * mean([row["CHAIRs"] for row in rows]),
        "CHAIRi": 100 * hallucinated / max(1, generated),
        "Len": mean([row["length"] for row in rows]),
        "generated_mentions": generated,
        "hallucinated_mentions": hallucinated,
        "unique_correct_objects": sum(row["unique_correct_objects"] for row in rows),
        "distinct_1": mean([row["distinct_1"] for row in rows]),
        "distinct_2": mean([row["distinct_2"] for row in rows]),
        "repeated_2gram_ratio": mean([row["repeated_2gram_ratio"] for row in rows]),
        "repeated_3gram_ratio": mean([row["repeated_3gram_ratio"] for row in rows]),
        "repeated_4gram_ratio": mean([row["repeated_4gram_ratio"] for row in rows]),
        "exact_cycle_hits": sum(row["exact_cycle"] for row in rows),
    }


def interval(values):
    values.sort()
    return [
        values[int(0.025 * len(values))],
        values[int(0.975 * len(values)) - 1],
    ]


def mcnemar_exact(new_errors, fixes):
    discordant = new_errors + fixes
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(new_errors, fixes) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def paired_comparison(baseline, candidate, bootstrap, seed):
    pairs = list(zip(baseline, candidate))
    rng = random.Random(seed)
    samples = {
        "CHAIRs": [],
        "CHAIRi": [],
        "Len": [],
        "unique_correct_objects": [],
        "repeated_3gram_ratio": [],
        "repeated_4gram_ratio": [],
    }
    transitions = {"00": 0, "01": 0, "10": 0, "11": 0}
    for left, right in pairs:
        transitions[f"{left['CHAIRs']}{right['CHAIRs']}"] += 1

    for _ in range(bootstrap):
        chosen = [rng.choice(pairs) for _ in pairs]
        left = [item[0] for item in chosen]
        right = [item[1] for item in chosen]
        left_agg = aggregate(left)
        right_agg = aggregate(right)
        for metric in samples:
            samples[metric].append(right_agg[metric] - left_agg[metric])

    return {
        "baseline": aggregate(baseline),
        "candidate": aggregate(candidate),
        "delta_candidate_minus_baseline": {
            metric: mean(values) for metric, values in samples.items()
        },
        "delta_95ci": {metric: interval(values) for metric, values in samples.items()},
        "transitions": transitions,
        "mcnemar_exact_p": mcnemar_exact(transitions["01"], transitions["10"]),
    }


def aggregate_diagnostics(rows, field):
    values = [row[field] for row in rows if field in row]
    if not values:
        return None
    totals = {}
    for value in values:
        for key, number in value.items():
            totals.setdefault(key, []).append(number)
    return {key: mean(numbers) for key, numbers in totals.items()}


def main():
    args = parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with open(args.ids_file) as source:
        image_ids = [int(image_id) for image_id in json.load(source)]

    raw_rows = {
        method: load_rows(os.path.join(args.result_dir, filename), image_ids)
        for method, filename in METHOD_FILES.items()
    }
    dataset = CHAIRBench(num_samples=len(image_ids), image_ids=image_ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _ = dataset.evaluator
    per_image = {
        method: [describe(dataset, row) for row in rows]
        for method, rows in raw_rows.items()
    }
    metrics = {method: aggregate(rows) for method, rows in per_image.items()}
    comparisons = {
        "vhr_to_scvhr": paired_comparison(
            per_image["vhr"], per_image["scvhr"], args.bootstrap, args.seed
        ),
        "vhr_to_carescvhr": paired_comparison(
            per_image["vhr"], per_image["carescvhr"], args.bootstrap, args.seed + 1
        ),
        "scvhr_to_carescvhr": paired_comparison(
            per_image["scvhr"], per_image["carescvhr"], args.bootstrap, args.seed + 2
        ),
        "cascvhr_to_carescvhr": paired_comparison(
            per_image["cascvhr"], per_image["carescvhr"], args.bootstrap, args.seed + 3
        ),
    }

    vhr = metrics["vhr"]
    sc = metrics["scvhr"]
    care = metrics["carescvhr"]
    care_vs_vhr = comparisons["vhr_to_carescvhr"]
    care_vs_sc = comparisons["scvhr_to_carescvhr"]
    correct_loss = max(0, vhr["unique_correct_objects"] - sc["unique_correct_objects"])
    recovered = care["unique_correct_objects"] - sc["unique_correct_objects"]

    gates = {
        "sc_replication_chairs_improves_at_least_3": sc["CHAIRs"] <= vhr["CHAIRs"] - 3,
        "vs_vhr": {
            "chairs_improves_at_least_8": care["CHAIRs"] <= vhr["CHAIRs"] - 8,
            "chairi_improves_at_least_2": care["CHAIRi"] <= vhr["CHAIRi"] - 2,
            "chairs_ci_upper_at_most_minus_1": care_vs_vhr["delta_95ci"]["CHAIRs"][1] <= -1,
            "unique_correct_retention_at_least_98pct": (
                care["unique_correct_objects"] / max(1, vhr["unique_correct_objects"]) >= 0.98
            ),
            "length_change_within_5pct": abs(care["Len"] / vhr["Len"] - 1) <= 0.05,
            "cycles_not_above_vhr": care["exact_cycle_hits"] <= vhr["exact_cycle_hits"],
        },
        "vs_scvhr": {
            "chairs_regression_at_most_2": care["CHAIRs"] <= sc["CHAIRs"] + 2,
            "chairi_regression_at_most_0_5": care["CHAIRi"] <= sc["CHAIRi"] + 0.5,
            "unique_correct_not_below_sc": care["unique_correct_objects"] >= sc["unique_correct_objects"],
            "recovers_half_of_sc_loss": correct_loss == 0 or recovered >= 0.5 * correct_loss,
            "repeated_3gram_not_above_sc": (
                care["repeated_3gram_ratio"] <= sc["repeated_3gram_ratio"]
            ),
            "repeated_4gram_not_above_sc": (
                care["repeated_4gram_ratio"] <= sc["repeated_4gram_ratio"]
            ),
            "cycles_not_above_sc": care["exact_cycle_hits"] <= sc["exact_cycle_hits"],
        },
    }
    gates["care_dev_pass"] = all(gates["vs_vhr"].values()) and all(
        gates["vs_scvhr"].values()
    )

    report = {
        "protocol": {
            "ids_file": args.ids_file,
            "samples": len(image_ids),
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "no_parameter_search": True,
        },
        "files": {
            method: os.path.join(args.result_dir, filename)
            for method, filename in METHOD_FILES.items()
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "diagnostics": {
            "cascvhr_alignment": aggregate_diagnostics(
                raw_rows["cascvhr"], "counterfactual_alignment_stats"
            ),
            "carescvhr_alignment": aggregate_diagnostics(
                raw_rows["carescvhr"], "counterfactual_alignment_stats"
            ),
            "carescvhr_stability": aggregate_diagnostics(
                raw_rows["carescvhr"], "stability_stats"
            ),
        },
        "gates": gates,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
