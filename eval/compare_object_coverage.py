import argparse
import contextlib
import json
import os
import random
import statistics

from datasets import CHAIRBench


METRICS = (
    "mentions",
    "unique_mentions",
    "correct_mentions",
    "unique_correct_mentions",
    "hallucinated_mentions",
    "unique_hallucinated_mentions",
    "repeated_mentions",
)


def load_rows(path):
    rows = [json.loads(line) for line in open(path)]
    return {int(row["question_id"]): row for row in rows}


def describe(dataset, row):
    gt_objects, node_words = dataset.eval_hal_words(row["question_id"], row["text"])
    correct = [word for word in node_words if word in gt_objects]
    hallucinated = [word for word in node_words if word not in gt_objects]
    unique_words = set(node_words)
    return {
        "mentions": len(node_words),
        "unique_mentions": len(unique_words),
        "correct_mentions": len(correct),
        "unique_correct_mentions": len(set(correct)),
        "hallucinated_mentions": len(hallucinated),
        "unique_hallucinated_mentions": len(set(hallucinated)),
        "repeated_mentions": len(node_words) - len(unique_words),
    }


def aggregate(rows):
    result = {metric: sum(row[metric] for row in rows) for metric in METRICS}
    result["images"] = len(rows)
    result["mean_mentions_per_image"] = result["mentions"] / len(rows)
    result["mean_unique_correct_per_image"] = result["unique_correct_mentions"] / len(rows)
    result["repeated_mention_rate"] = result["repeated_mentions"] / max(1, result["mentions"])
    result["correct_mention_rate"] = result["correct_mentions"] / max(1, result["mentions"])
    return result


def interval(values):
    values.sort()
    return [
        values[int(0.025 * len(values))],
        values[int(0.975 * len(values)) - 1],
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Compare normalized COCO-object coverage without regenerating captions."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--ids_file", required=True)
    parser.add_argument("--bootstrap", default=20000, type=int)
    parser.add_argument("--seed", default=20260716, type=int)
    args = parser.parse_args()

    image_ids = [int(image_id) for image_id in json.load(open(args.ids_file))]
    baseline_rows = load_rows(args.baseline)
    candidate_rows = load_rows(args.candidate)
    expected = set(image_ids)
    if set(baseline_rows) != expected or set(candidate_rows) != expected:
        raise SystemExit("Baseline/candidate image IDs do not exactly match ids_file")

    dataset = CHAIRBench(num_samples=len(image_ids), image_ids=image_ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _ = dataset.evaluator

    pairs = [
        (describe(dataset, baseline_rows[image_id]), describe(dataset, candidate_rows[image_id]))
        for image_id in image_ids
    ]
    baseline_metrics = aggregate([baseline for baseline, _ in pairs])
    candidate_metrics = aggregate([candidate for _, candidate in pairs])

    rng = random.Random(args.seed)
    delta_samples = {metric: [] for metric in METRICS}
    for _ in range(args.bootstrap):
        sample = [rng.choice(pairs) for _ in pairs]
        for metric in METRICS:
            delta_samples[metric].append(
                statistics.mean(candidate[metric] - baseline[metric] for baseline, candidate in sample)
            )

    result = {
        "baseline_file": args.baseline,
        "candidate_file": args.candidate,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "candidate_to_baseline_ratio": {
            metric: candidate_metrics[metric] / max(1, baseline_metrics[metric])
            for metric in METRICS
        },
        "paired_mean_delta_per_image": {
            metric: statistics.mean(delta_samples[metric]) for metric in METRICS
        },
        "paired_mean_delta_95ci": {
            metric: interval(delta_samples[metric]) for metric in METRICS
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
