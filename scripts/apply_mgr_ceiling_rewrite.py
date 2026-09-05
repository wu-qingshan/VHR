#!/usr/bin/env python3
"""Apply oracle acceptance rules to operator C candidates and finalize E0.1."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy

from datasets import CHAIRBench
from qwen_sear import extract_caption
from qwen_vera import apply_sentence_revisions, exact_cycle, sanitize_extraction


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


def atomic_jsonl_new(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quick_metrics(per_image):
    rows = list(per_image.values())
    hallucinated = sum(row["hallucinated_mentions"] for row in rows)
    generated = sum(row["generated_mentions"] for row in rows)
    return {
        "CHAIRs": 100.0 * sum(row["CHAIRs"] for row in rows) / max(1, len(rows)),
        "CHAIRi": 100.0 * hallucinated / max(1, generated),
        "Len": sum(row["length"] for row in rows) / max(1, len(rows)),
        "hallucinated_mentions": hallucinated,
        "generated_mentions": generated,
        "unique_correct_objects": sum(len(row["unique_correct"]) for row in rows),
        "unique_hallucinated_objects": sum(len(row["unique_hallucinated"]) for row in rows),
        "exact_cycle_images": sum(1 for row in rows if row["exact_cycle"]),
    }


def describe_caption(dataset, image_id, text):
    gt_objects, node_words = dataset.eval_hal_words(image_id, text)
    correct = [word for word in node_words if word in gt_objects]
    hallucinated = [word for word in node_words if word not in gt_objects]
    return {
        "CHAIRs": int(bool(hallucinated)),
        "hallucinated_mentions": len(hallucinated),
        "generated_mentions": len(node_words),
        "length": len(text.split()),
        "unique_correct": set(correct),
        "unique_hallucinated": set(hallucinated),
        "exact_cycle": exact_cycle(text) is not None,
    }


def bootstrap_delta(base, candidate, draws, seed):
    import random

    keys = sorted(base, key=int)
    pairs = [(base[key], candidate[key]) for key in keys]
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [rng.choice(pairs) for _ in pairs]
        values.append(
            100.0
            * sum(right["CHAIRs"] - left["CHAIRs"] for left, right in sample)
            / len(sample)
        )
    values.sort()
    return {
        "point": 100.0
        * sum(right["CHAIRs"] - left["CHAIRs"] for left, right in pairs)
        / len(pairs),
        "95ci": [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]],
        "probability_below_zero": sum(value < 0 for value in values) / len(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", default="experiment/qwen_vera_dev_chair_ids.json")
    parser.add_argument("--captions", default="qwen_vera_dev_results/vanilla.json")
    parser.add_argument(
        "--extractions", default="qwen_vera_dev_results/source/extractions.json"
    )
    parser.add_argument("--tasks", default="mgr_ceiling_results/operatorC_tasks.json")
    parser.add_argument(
        "--candidates", default="mgr_ceiling_results/rewrite_candidates.jsonl"
    )
    parser.add_argument("--oracle", default="mgr_ceiling_results/oracle_triggers.json")
    parser.add_argument(
        "--partial", default="mgr_ceiling_results/ceiling_analysis.json"
    )
    parser.add_argument("--output_dir", default="mgr_ceiling_results")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026071701)
    args = parser.parse_args()

    ids = [int(value) for value in json.load(open(args.ids_file))]
    captions = load_jsonl(args.captions)
    extractions = json.load(open(args.extractions))
    tasks = {row["revision_id"]: row for row in json.load(open(args.tasks))}
    candidates = {row["revision_id"]: row for row in load_jsonl(args.candidates)}
    oracle = {row["image_id"]: row for row in json.load(open(args.oracle))}
    if [int(row["question_id"]) for row in captions] != ids:
        raise ValueError("caption order does not match the frozen dev id list")
    if [int(row["image_id"]) for row in extractions] != ids:
        raise ValueError("extraction order does not match the frozen dev id list")
    if set(tasks) != set(candidates):
        raise ValueError("candidate file does not cover exactly the operator C tasks")

    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        evaluator = dataset.evaluator
    mscoco_objects = set(evaluator.mscoco_objects)

    nlp = spacy.load("en_core_web_sm")

    audits = []
    replacements_by_image = {}
    for revision_id, task in tasks.items():
        row = candidates[revision_id]
        if hashlib.sha256(task["source_sentence"].encode()).hexdigest() != row[
            "source_sentence_sha256"
        ]:
            raise ValueError(f"source sentence changed for {revision_id}")
        image_id = int(task["image_id"])
        hallucinated_nodes = set(oracle[image_id]["hallucinated_nodes"])
        candidate_text = row["candidate"]["text"]
        extraction = sanitize_extraction(extract_caption(candidate_text, nlp))
        candidate_roots = {c["root_lemma"] for c in extraction["candidates"]}
        hallucinated_roots = sorted(
            root
            for root in candidate_roots
            if root in mscoco_objects
            and evaluator.inverse_synonym_dict[root] in hallucinated_nodes
        )
        source_words = len(task["source_sentence"].split())
        candidate_words = len(candidate_text.split())
        checks = {
            "exactly_one_nonempty_sentence": len(extraction["sentences"]) == 1
            and bool(candidate_text.strip()),
            "all_protected_roots_retained": set(task["protected_roots"])
            <= candidate_roots,
            "all_trigger_roots_removed": not (
                set(task["trigger_roots"]) & candidate_roots
            ),
            "no_chair_hallucinated_root": not hallucinated_roots,
            "length_between_half_and_1_25": 0.5 * source_words
            <= candidate_words
            <= 1.25 * source_words,
            "no_exact_cycle": exact_cycle(candidate_text) is None,
        }
        passed = all(checks.values())
        audits.append(
            {
                "revision_id": revision_id,
                "image_id": image_id,
                "sentence_index": int(task["sentence_index"]),
                "candidate_text": candidate_text,
                "candidate_roots": sorted(candidate_roots),
                "hallucinated_roots": hallucinated_roots,
                "checks": checks,
                "passed": passed,
                "generation_elapsed_seconds": row["candidate"]["elapsed_seconds"],
            }
        )
        if passed:
            replacements_by_image.setdefault(str(image_id), {})[
                str(int(task["sentence_index"]))
            ] = candidate_text

    operator_c_rows = []
    per_image_c = {}
    for caption, extraction in zip(captions, extractions):
        image_id = int(caption["question_id"])
        text = caption["text"]
        if hashlib.sha256(text.encode()).hexdigest() != extraction["text_sha256"]:
            raise ValueError(f"source caption changed for {image_id}")
        replacements = replacements_by_image.get(str(image_id), {})
        if replacements:
            revised = apply_sentence_revisions(text, extraction, replacements)
        else:
            revised = text
        operator_c_rows.append(
            {"question_id": image_id, "text": revised, "method": "oracleC"}
        )
        per_image_c[str(image_id)] = describe_caption(dataset, image_id, revised)

    output_dir = Path(args.output_dir)
    atomic_json_new(output_dir / "operatorC_audits.json", audits)
    atomic_jsonl_new(output_dir / "operatorC.jsonl", operator_c_rows)

    partial = json.load(open(args.partial))
    metrics = dict(partial["metrics"])
    metrics["operatorC"] = quick_metrics(per_image_c)
    summary = dict(partial)
    summary["metrics"] = metrics
    summary["operatorC_pending"] = False
    summary["operatorC"] = {
        "tasks": len(tasks),
        "accepted": len(replacements_by_image) and sum(
            len(v) for v in replacements_by_image.values()
        ),
        "acceptance_rate": sum(len(v) for v in replacements_by_image.values())
        / max(1, len(tasks)),
        "mean_generation_seconds": sum(a["generation_elapsed_seconds"] for a in audits)
        / max(1, len(audits)),
        "check_failure_counts": {
            name: sum(1 for a in audits if not a["checks"][name])
            for name in next(iter(audits))["checks"]
        },
        "inputs": {
            "candidates": args.candidates,
            "candidates_sha256": sha256_file(args.candidates),
        },
    }
    summary["paired_bootstrap_vs_vanilla"] = dict(
        partial["paired_bootstrap_vs_vanilla"]
    )
    summary["paired_bootstrap_vs_vanilla"]["operatorC"] = bootstrap_delta(
        partial["per_image"]["vanilla"],
        {
            key: {
                "CHAIRs": row["CHAIRs"],
                "hallucinated_mentions": row["hallucinated_mentions"],
                "generated_mentions": row["generated_mentions"],
                "length": row["length"],
                "unique_correct": sorted(row["unique_correct"]),
            }
            for key, row in per_image_c.items()
        },
        args.bootstrap,
        args.bootstrap_seed,
    )
    summary["per_image"] = dict(partial["per_image"])
    summary["per_image"]["operatorC"] = {
        key: {
            "CHAIRs": row["CHAIRs"],
            "hallucinated_mentions": row["hallucinated_mentions"],
            "generated_mentions": row["generated_mentions"],
            "length": row["length"],
            "unique_correct": sorted(row["unique_correct"]),
        }
        for key, row in per_image_c.items()
    }
    atomic_json_new(output_dir / "ceiling_analysis_final.json", summary)
    printable = {k: summary[k] for k in summary if k not in {"per_image", "operator_infos"}}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
