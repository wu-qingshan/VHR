#!/usr/bin/env python3
"""Compute the preregistered Qwen object-verifier feasibility gate."""

import argparse
import json
import random
import statistics
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="qwen_object_evidence_results/verifier.json")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", default="qwen_object_evidence_results/analysis.json")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.input) if line.strip()]
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    predictions = [int(score >= 0.5) for score in scores]
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]

    rng = random.Random(args.seed)
    gaps = []
    for _ in range(args.bootstrap):
        pos = [rng.choice(positives) for _ in positives]
        neg = [rng.choice(negatives) for _ in negatives]
        gaps.append(mean(pos) - mean(neg))
    gaps.sort()
    gap_ci = [gaps[int(0.025 * len(gaps))], gaps[int(0.975 * len(gaps)) - 1]]

    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    metrics = {
        "queries": len(rows),
        "positive": len(positives),
        "negative": len(negatives),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_tn_fp_fn_tp": [int(matrix[0][0]), int(matrix[0][1]), int(matrix[1][0]), int(matrix[1][1])],
        "positive_score_mean": mean(positives),
        "positive_score_median": statistics.median(positives),
        "negative_score_mean": mean(negatives),
        "negative_score_median": statistics.median(negatives),
        "mean_gap": mean(positives) - mean(negatives),
        "mean_gap_95ci": gap_ci,
        "mean_seconds_per_query": mean([row["elapsed_seconds"] for row in rows]),
        "mean_peak_allocated_gib": mean([row["peak_allocated_gib"] for row in rows]),
    }
    gates = {
        "queries_at_least_500": bool(metrics["queries"] >= 500),
        "negative_at_least_60": bool(metrics["negative"] >= 60),
        "roc_auc_at_least_0_75": bool(metrics["roc_auc"] >= 0.75),
        "balanced_accuracy_at_least_0_70": bool(metrics["balanced_accuracy"] >= 0.70),
        "mean_gap_ci_lower_above_zero": bool(gap_ci[0] > 0),
    }
    errors = {
        "false_positive_examples": [
            row for row in rows if row["label"] == 0 and row["prediction"] == 1
        ][:30],
        "false_negative_examples": [
            row for row in rows if row["label"] == 1 and row["prediction"] == 0
        ][:30],
    }
    report = {
        "metrics": metrics,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": "implement_oer_revision" if all(gates.values()) else "stop_self_verification",
        "errors": errors,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("metrics", "gates", "all_gates_passed", "decision")}, indent=2))


if __name__ == "__main__":
    main()
