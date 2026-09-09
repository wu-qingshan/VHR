#!/usr/bin/env python3
"""Stage-1 MGR verifier feasibility analysis (the only CHAIR-importing step).

Reference labels come from the frozen CHAIR evaluator; the pipeline itself never
saw them. Gate verdicts require the completed E0.2 human annotation CSV.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench
from qwen_mgr import combine_scores, is_triggered


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def atomic_json_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roc_auc(pairs):
    """pairs: (score, label) with label 1 = hallucinated, lower score = hallucinated."""

    positives = [score for score, label in pairs if label == 1]
    negatives = [score for score, label in pairs if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos < neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def pr_auc(pairs):
    """Average precision: positives ranked by ascending score (low = hallucinated)."""

    ranked = sorted(pairs, key=lambda row: row[0])
    total_positive = sum(label for _, label in ranked)
    if total_positive == 0:
        return None
    hits = 0
    accumulated = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            hits += 1
            accumulated += hits / index
    return accumulated / total_positive


def read_e02_precision(csv_path):
    labels = []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("label") or "").strip()
            if value:
                labels.append(int(value))
    if not labels:
        return None, 0
    return sum(1 for value in labels if value == 1) / len(labels), len(labels)


def row_metrics(mentions):
    labeled = [m for m in mentions if m["label"] in (0, 1)]
    positives = [m for m in labeled if m["label"] == 1]
    triggered = [m for m in mentions if m["triggered"]]
    triggered_labeled = [m for m in triggered if m["label"] in (0, 1)]
    triggered_hallucinated = [m for m in triggered if m["label"] == 1]
    return {
        "mentions": len(mentions),
        "labeled_mentions": len(labeled),
        "hallucinated_mentions": len(positives),
        "triggered": len(triggered),
        "triggered_hallucinated": len(triggered_hallucinated),
        "precision_chair": len(triggered_hallucinated)
        / max(1, len(triggered_labeled)),
        "precision_all": len(triggered_hallucinated) / max(1, len(triggered)),
        "unlabeled_trigger_share": (
            sum(1 for m in triggered if m["label"] is None) / max(1, len(triggered))
        ),
        "recall": len(triggered_hallucinated) / max(1, len(positives)),
        "roc_auc": roc_auc([(m["score"], m["label"]) for m in labeled]),
        "pr_auc": pr_auc([(m["score"], m["label"]) for m in labeled]),
        "s_whole_roc_auc": roc_auc([(m["s_whole"], m["label"]) for m in labeled]),
        "ungrounded_rate": sum(1 for m in mentions if m["region_score"] is None)
        / max(1, len(mentions)),
    }


def h_coverage(dataset, inventory, mentions_by_image):
    evaluator = dataset.evaluator
    covered_total = 0
    hallucinated_total = 0
    for image in inventory:
        image_id = image["image_id"]
        gt = evaluator.imid_to_objects[image_id]
        caption = caption_cache[image_id]
        _, node_words, _, _ = evaluator.caption_to_words(caption)
        hallucinated = {w for w in node_words if w not in gt}
        if not hallucinated:
            continue
        covered = set()
        for mention in mentions_by_image.get(image_id, []):
            if not mention["triggered"]:
                continue
            covered |= set(mention["node_words"]) & hallucinated
        covered_total += len(covered)
        hallucinated_total += len(hallucinated)
    return covered_total / max(1, hallucinated_total)


def caption_lookup():
    captions = {}
    for path in (
        "qwen_dev_results/vanilla.json",
        "qwen_vera_dev_results/vanilla.json",
    ):
        for row in load_jsonl(path):
            captions[int(row["question_id"])] = row["text"]
    return captions


caption_cache = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="mgr_verifier_results")
    parser.add_argument(
        "--e02_csv", default="mgr_ceiling_results/e02_annotation_form.csv"
    )
    parser.add_argument("--output", default="mgr_verifier_results/analysis.json")
    args = parser.parse_args()
    results = Path(args.results_dir)

    inventory = json.load(open(results / "inventory.json"))
    queries = json.load(open(results / "whole_queries.json"))["queries"]

    whole_new = {}
    for path in sorted(results.glob("whole-verifier-chunk*.jsonl")):
        for row in load_jsonl(path):
            whole_new[row["query_id"]] = row
    region = {}
    verifier_checks = {}
    for path in sorted(results.glob("region-chunk*.jsonl")):
        for row in load_jsonl(path):
            if row.get("query_type") == "grounding":
                region[row["query_id"]] = row
            else:
                verifier_checks[f"{path.name}:{row['query_type']}"] = {
                    "object": row.get("object"),
                    "image_id": row.get("image_id"),
                    "score": row.get("score"),
                }

    missing_whole = [
        q["query_id"] for q in queries if q["reused_query_id"] is None and q["query_id"] not in whole_new
    ]
    missing_region = [q["query_id"] for q in queries if q["query_id"] not in region]
    if missing_whole or missing_region:
        print(
            json.dumps(
                {
                    "incomplete": True,
                    "missing_whole": len(missing_whole),
                    "missing_region": len(missing_region),
                    "missing_examples": (missing_whole + missing_region)[:5],
                }
            )
        )
        return

    e02_precision, e02_labels = read_e02_precision(args.e02_csv)
    e02_complete = e02_labels >= 38

    ids = []
    for path in (
        "experiment/qwen_dev_chair_ids.json",
        "experiment/qwen_vera_dev_chair_ids.json",
    ):
        ids += [int(value) for value in json.load(open(path))]
    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        evaluator = dataset.evaluator
    mscoco_objects = set(evaluator.mscoco_objects)
    global caption_cache
    caption_cache = caption_lookup()

    spacy_verifier = {}
    materialized_elapsed = {}
    for path in (
        "qwen_sear_dev_results/verifier.json",
        "qwen_vera_dev_results/source/verifier.json",
    ):
        for row in load_jsonl(path):
            spacy_verifier[(int(row["image_id"]), row["query_id"])] = row
            materialized_elapsed[
                (int(row["image_id"]), row["object"].strip().lower())
            ] = row["elapsed_seconds"]

    mention_records = []
    for query in queries:
        image_id = int(query["image_id"])
        gt = evaluator.imid_to_objects[image_id]
        whole_row = whole_new.get(query["query_id"])
        s_whole = (
            float(whole_row["score"])
            if whole_row
            else float(query["reused_score"])
        )
        region_row = region[query["query_id"]]
        region_score = (
            float(region_row["region_score"])
            if region_row["region_score"] is not None
            else None
        )
        score = combine_scores(s_whole, region_score)
        mention_string = query["object"]
        _, node_words, _, _ = evaluator.caption_to_words(mention_string)
        node_words = list(node_words)
        if not node_words:
            label = None
        elif any(word not in gt for word in node_words):
            label = 1
        else:
            label = 0
        mention_records.append(
            {
                "query_id": query["query_id"],
                "set": query["set"],
                "image_id": image_id,
                "object": mention_string,
                "object_key": query["object_key"],
                "source": query["source"],
                "node_words": node_words,
                "label": label,
                "s_whole": s_whole,
                "region_score": region_score,
                "ungrounded_reason": region_row["ungrounded_reason"],
                "score": score,
                "triggered": is_triggered(score),
                "whole_elapsed_seconds": (
                    whole_row["elapsed_seconds"]
                    if whole_row
                    else materialized_elapsed.get((image_id, query["object_key"]))
                ),
                "grounding_elapsed_seconds": region_row["grounding_elapsed_seconds"],
                "region_elapsed_seconds": region_row["region_elapsed_seconds"],
                "s_whole_only_triggered": is_triggered(s_whole),
            }
        )

    empty_selflist_images = {
        row["image_id"] for row in inventory if row["caption_selflist_empty"]
    }
    rows = {}
    rows["selflist_only"] = [m for m in mention_records if m["source"] == "selflist"]
    rows["branch_S"] = [
        m
        for m in mention_records
        if m["source"] == "selflist" or m["image_id"] in empty_selflist_images
    ]
    rows["branch_U"] = list(mention_records)

    selflist_elapsed = {}
    for path in sorted(results.glob("selflist-*.jsonl")):
        if "chunk" in path.name:
            continue
        for row in load_jsonl(path):
            selflist_elapsed[int(row["question_id"])] = row["elapsed_seconds"]

    metrics_rows = {}
    mentions_by_image = {}
    for name, mentions in rows.items():
        if mentions is None:
            continue
        metrics_rows[name] = row_metrics(mentions)
        mentions_by_image = {}
        for mention in mentions:
            mentions_by_image.setdefault(mention["image_id"], []).append(mention)
        metrics_rows[name]["h_coverage"] = h_coverage(
            dataset, inventory, mentions_by_image
        )

    # spacy_only: VERA-equivalent inventory scored by s_whole from materialized runs.
    spacy_records = []
    for image in inventory:
        gt = evaluator.imid_to_objects[image["image_id"]]
        for candidate in image["spacy_candidates"]:
            row = spacy_verifier[(image["image_id"], candidate["query_id"])]
            score = float(row["score"])
            mention_string = candidate["phrase"]
            _, node_words, _, _ = evaluator.caption_to_words(mention_string)
            node_words = list(node_words)
            if not node_words:
                label = None
            elif any(word not in gt for word in node_words):
                label = 1
            else:
                label = 0
            spacy_records.append(
                {
                    "query_id": candidate["query_id"],
                    "set": image["set"],
                    "image_id": image["image_id"],
                    "object": mention_string,
                    "source": "spacy",
                    "node_words": node_words,
                    "label": label,
                    "s_whole": score,
                    "region_score": None,
                    "ungrounded_reason": "not_run",
                    "score": score,
                    "triggered": is_triggered(score),
                    "whole_elapsed_seconds": row["elapsed_seconds"],
                    "grounding_elapsed_seconds": None,
                    "region_elapsed_seconds": None,
                    "s_whole_only_triggered": is_triggered(score),
                }
            )
    metrics_rows["spacy_only"] = row_metrics(spacy_records)
    mentions_by_image = {}
    for mention in spacy_records:
        mentions_by_image.setdefault(mention["image_id"], []).append(mention)
    metrics_rows["spacy_only"]["h_coverage"] = h_coverage(
        dataset, inventory, mentions_by_image
    )

    primary = None
    branch_selection = {
        "e02_precision": e02_precision,
        "e02_labels": e02_labels,
        "complete": e02_complete,
    }
    if e02_complete and e02_precision is not None:
        if e02_precision < 0.3:
            primary = "branch_S"
        else:
            primary = "branch_U"
        branch_selection["selected_branch"] = primary

    cost = {}
    for name, mentions in (
        ("branch_U", mention_records),
        ("selflist_only", rows["selflist_only"]),
    ):
        per_image = {}
        for mention in mentions:
            image_id = mention["image_id"]
            total = per_image.get(image_id, 0.0)
            if mention["whole_elapsed_seconds"] is not None:
                total += mention["whole_elapsed_seconds"]
            if mention["s_whole"] <= 0.1:
                total += mention["grounding_elapsed_seconds"] or 0.0
                total += mention["region_elapsed_seconds"] or 0.0
            per_image[image_id] = total
        cost[name] = {
            "mean_verifier_query_seconds_per_image": sum(per_image.values())
            / max(1, len(per_image)),
            "images": len(per_image),
        }
    pipeline_images = set(selflist_elapsed)
    cost["selflist_mean_seconds_per_image"] = sum(selflist_elapsed.values()) / max(
        1, len(pipeline_images)
    )

    gates = None
    if primary is not None:
        row = metrics_rows[primary]
        gates = {
            "g1_sample_size": {
                "labeled_mentions": row["labeled_mentions"],
                "hallucinated_mentions": row["hallucinated_mentions"],
                "passed": row["labeled_mentions"] >= 1500
                and row["hallucinated_mentions"] >= 200,
            },
            "g2_precision_chair": {
                "value": row["precision_chair"],
                "passed": row["precision_chair"] >= 0.5,
            },
            "g3_recall": {"value": row["recall"], "passed": row["recall"] >= 0.6},
            "g4_roc_auc": {
                "value": row["roc_auc"],
                "passed": row["roc_auc"] is not None and row["roc_auc"] >= 0.85,
            },
            "g5_verifier_cost": {
                "mean_seconds_per_image": cost["branch_U"][
                    "mean_verifier_query_seconds_per_image"
                ],
                "passed": cost["branch_U"][
                    "mean_verifier_query_seconds_per_image"
                ]
                <= 8.0,
            },
        }
        gates["all_passed"] = all(item["passed"] for item in gates.values())

    hallucinated_examples = [
        {
            "query_id": m["query_id"],
            "object": m["object"],
            "ungrounded_reason": m["ungrounded_reason"],
            "region_score": m["region_score"],
            "s_whole": m["s_whole"],
            "triggered": m["triggered"],
        }
        for m in mention_records
        if m["label"] == 1
    ][:40]
    false_trigger_examples = [
        {
            "query_id": m["query_id"],
            "object": m["object"],
            "node_words": m["node_words"],
            "s_whole": m["s_whole"],
            "region_score": m["region_score"],
            "ungrounded_reason": m["ungrounded_reason"],
        }
        for m in mention_records
        if m["triggered"] and m["label"] == 0
    ][:40]

    report = {
        "posthoc_reference_only": True,
        "branch_selection": branch_selection,
        "rows": metrics_rows,
        "gate_verdict": gates,
        "cost": cost,
        "verifier_leak_checks": verifier_checks,
        "counts": {
            "queries": len(queries),
            "reused_whole": sum(
                1 for q in queries if q["reused_query_id"] is not None
            ),
        },
        "hallucinated_examples": hallucinated_examples,
        "false_trigger_examples": false_trigger_examples,
        "inputs": {
            "inventory_sha256": sha256_file(results / "inventory.json"),
            "e02_csv_sha256": sha256_file(args.e02_csv),
        },
    }
    atomic_json_new(args.output, report)
    compact = {
        key: report[key]
        for key in ("branch_selection", "rows", "gate_verdict", "cost", "counts")
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
