import argparse
import contextlib
import json
import os
import random
import statistics

from datasets import CHAIRBench
from eval.compare_object_coverage import METRICS, aggregate, describe, interval, load_rows


def main():
    parser = argparse.ArgumentParser(
        description="Pool normalized object coverage across disjoint baseline/candidate/ID groups."
    )
    parser.add_argument(
        "--group",
        nargs=3,
        action="append",
        metavar=("BASELINE", "CANDIDATE", "IDS_FILE"),
        required=True,
    )
    parser.add_argument("--bootstrap", default=50000, type=int)
    parser.add_argument("--seed", default=20260717, type=int)
    args = parser.parse_args()

    groups = []
    all_ids = []
    seen_ids = set()
    for baseline_path, candidate_path, ids_path in args.group:
        image_ids = [int(image_id) for image_id in json.load(open(ids_path))]
        overlap = seen_ids & set(image_ids)
        if overlap:
            raise SystemExit(f"Groups are not disjoint; first overlaps: {sorted(overlap)[:10]}")
        seen_ids.update(image_ids)
        all_ids.extend(image_ids)
        baseline_rows = load_rows(baseline_path)
        candidate_rows = load_rows(candidate_path)
        if set(baseline_rows) != set(image_ids) or set(candidate_rows) != set(image_ids):
            raise SystemExit(f"Result IDs do not match {ids_path}")
        groups.append((baseline_path, candidate_path, image_ids, baseline_rows, candidate_rows))

    dataset = CHAIRBench(num_samples=len(all_ids), image_ids=all_ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _ = dataset.evaluator

    pairs = []
    group_summaries = []
    for baseline_path, candidate_path, image_ids, baseline_rows, candidate_rows in groups:
        group_pairs = [
            (describe(dataset, baseline_rows[image_id]), describe(dataset, candidate_rows[image_id]))
            for image_id in image_ids
        ]
        pairs.extend(group_pairs)
        group_summaries.append(
            {
                "baseline": baseline_path,
                "candidate": candidate_path,
                "samples": len(image_ids),
                "baseline_metrics": aggregate([baseline for baseline, _ in group_pairs]),
                "candidate_metrics": aggregate([candidate for _, candidate in group_pairs]),
            }
        )

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
        "groups": group_summaries,
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
