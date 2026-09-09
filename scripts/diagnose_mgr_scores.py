#!/usr/bin/env python3
"""Explanation-only diagnosis: score distributions for hallucinated vs correct mentions."""

import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench
from qwen_mgr import combine_scores, is_triggered


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="mgr_verifier_results")
    args = parser.parse_args()
    results = os.path.abspath(args.results_dir)

    queries = json.load(open(os.path.join(results, "whole_queries.json")))["queries"]
    whole_new = {}
    for path in sorted(
        f for f in os.listdir(results) if f.startswith("whole-verifier-chunk")
    ):
        for row in load_jsonl(os.path.join(results, path)):
            whole_new[row["query_id"]] = row
    region = {}
    for path in sorted(f for f in os.listdir(results) if f.startswith("region-chunk")):
        for row in load_jsonl(os.path.join(results, path)):
            if row.get("query_type") == "grounding":
                region[row["query_id"]] = row

    ids = []
    for id_path in (
        "experiment/qwen_dev_chair_ids.json",
        "experiment/qwen_vera_dev_chair_ids.json",
    ):
        ids += [int(value) for value in json.load(open(id_path))]
    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        evaluator = dataset.evaluator

    buckets = {"hallucinated": [], "correct": []}
    bands = {"hallucinated": {}, "correct": {}}
    examples = {"hallucinated_high_score": [], "correct_low_score": []}
    for query in queries:
        image_id = int(query["image_id"])
        gt = evaluator.imid_to_objects[image_id]
        mention = query["object"]
        _, node_words, _, _ = evaluator.caption_to_words(mention)
        if not node_words:
            continue
        hallucinated = any(word not in gt for word in node_words)
        key = "hallucinated" if hallucinated else "correct"
        whole_row = whole_new.get(query["query_id"])
        s_whole = float(whole_row["score"]) if whole_row else float(query["reused_score"])
        region_row = region[query["query_id"]]
        region_score = (
            float(region_row["region_score"])
            if region_row["region_score"] is not None
            else None
        )
        score = combine_scores(s_whole, region_score)
        buckets[key].append((s_whole, score))
        band = (
            "<=0.1"
            if score <= 0.1
            else "<=0.3"
            if score <= 0.3
            else "<=0.5"
            if score <= 0.5
            else "<=0.9"
            if score <= 0.9
            else ">0.9"
        )
        bands[key][band] = bands[key].get(band, 0) + 1
        if key == "hallucinated" and s_whole > 0.9 and len(examples["hallucinated_high_score"]) < 12:
            examples["hallucinated_high_score"].append(
                {
                    "image_id": image_id,
                    "mention": mention,
                    "node_words": list(node_words),
                    "s_whole": round(s_whole, 4),
                    "ungrounded_reason": region_row["ungrounded_reason"],
                    "region_score": region_score,
                }
            )
        if key == "correct" and is_triggered(score) and len(examples["correct_low_score"]) < 12:
            examples["correct_low_score"].append(
                {
                    "image_id": image_id,
                    "mention": mention,
                    "node_words": list(node_words),
                    "s_whole": round(s_whole, 4),
                    "ungrounded_reason": region_row["ungrounded_reason"],
                    "region_score": region_score,
                }
            )

    report = {"posthoc_explanation_only": True, "must_not_tune_current_dev": True}
    for key in ("hallucinated", "correct"):
        values = buckets[key]
        whole = sorted(v[0] for v in values)
        combined = sorted(v[1] for v in values)
        report[key] = {
            "n": len(values),
            "s_whole_median": whole[len(whole) // 2],
            "s_whole_below_0.1": sum(v <= 0.1 for v in whole),
            "combined_median": combined[len(combined) // 2],
            "combined_below_0.1": sum(v <= 0.1 for v in combined),
            "combined_bands": bands[key],
        }
    report["examples"] = examples
    output = os.path.join(results, "score_diagnosis.json")
    if os.path.exists(output):
        raise FileExistsError(output)
    open(output, "w").write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("hallucinated", "correct")}, indent=2))


if __name__ == "__main__":
    main()
