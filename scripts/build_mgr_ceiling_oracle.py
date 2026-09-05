#!/usr/bin/env python3
"""Build MGR ceiling oracle triggers and offline operators A/B (E0.1).

Oracle construction uses CHAIR ground-truth labels for explanation-only
ceiling measurement (posthoc_explanation_only=true,
must_not_tune_current_dev=true). It never feeds any method component.
"""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench
from qwen_sear import filter_caption
from qwen_vera import format_revision_prompt


PUNCTUATION_FIX = r"[ \t]+([,.;:!?])"
REPEATED_COMMA = re.compile(r",[ \t]*,")


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


def cleanup_operator_a(text):
    """The only preregistered cleanup rules for mention deletion."""

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(PUNCTUATION_FIX, r"\1", text)
    while True:
        replaced = REPEATED_COMMA.sub(",", text)
        if replaced == text:
            break
        text = replaced
    return text


def apply_operator_a(text, extraction, triggered_roots):
    """Delete every triggered occurrence span, replacing it with one space."""

    spans = []
    for candidate in extraction["candidates"]:
        if candidate["root_lemma"] not in triggered_roots:
            continue
        for occurrence in candidate["occurrences"]:
            start = int(occurrence["start_char"])
            end = int(occurrence["end_char"])
            if text[start:end] != occurrence["phrase"]:
                raise AssertionError(
                    f"occurrence span mismatch at [{start},{end}): "
                    f"{text[start:end]!r} != {occurrence['phrase']!r}"
                )
            spans.append((start, end, occurrence["phrase"]))
    if not spans:
        return text, {"deleted_occurrences": 0, "fallback_used": False}
    spans.sort(key=lambda row: row[0])
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            raise AssertionError(f"overlapping deletion spans: {left} {right}")
    pieces = []
    cursor = 0
    for start, end, _ in spans:
        pieces.append(text[cursor:start])
        pieces.append(" ")
        cursor = end
    pieces.append(text[cursor:])
    revised = cleanup_operator_a("".join(pieces))
    if not revised.strip():
        return text, {"deleted_occurrences": len(spans), "fallback_used": True}
    return revised, {"deleted_occurrences": len(spans), "fallback_used": False}


def apply_operator_b(text, extraction, triggered_roots):
    """Sentence deletion via the frozen SEAR filter with oracle scores."""

    scores = {}
    for candidate in extraction["candidates"]:
        query_id = candidate["query_id"]
        if query_id in scores:
            raise ValueError(f"duplicate query id {query_id}")
        scores[query_id] = 0.0 if candidate["root_lemma"] in triggered_roots else 1.0
    filtered, info = filter_caption(text, extraction, scores, threshold=0.1)
    return filtered, {
        "deleted_sentence_indices": info["deleted_sentence_indices"],
        "fallback_used": info["fallback_used"],
    }


def build_operator_c_tasks(image_id, text, extraction, triggered_roots):
    """One base-rewrite task per sentence containing a triggered occurrence."""

    sentence_roots = {
        sentence["index"]: [] for sentence in extraction["sentences"]
    }
    for candidate in extraction["candidates"]:
        for occurrence in candidate["occurrences"]:
            sentence_roots[int(occurrence["sentence_index"])].append(
                (candidate["root_lemma"], occurrence["phrase"])
            )
    tasks = []
    for sentence in extraction["sentences"]:
        index = int(sentence["index"])
        roots = sentence_roots[index]
        triggers = [root for root, _ in roots if root in triggered_roots]
        if not triggers:
            continue
        trigger_roots = sorted(set(triggers))
        trigger_phrases = []
        seen = set()
        for root, phrase in roots:
            if root in triggered_roots and phrase not in seen:
                seen.add(phrase)
                trigger_phrases.append(phrase)
        protected = sorted({root for root, _ in roots} - set(trigger_roots))
        start = int(sentence["start_char"])
        end = int(sentence["end_char"])
        source_sentence = text[start:end]
        if source_sentence != sentence["text"]:
            raise AssertionError("source sentence span is not exact")
        tasks.append(
            {
                "revision_id": f"{image_id}:{index:03d}",
                "image_id": image_id,
                "sentence_index": index,
                "source_sentence": source_sentence,
                "source_sentence_sha256": hashlib.sha256(
                    source_sentence.encode()
                ).hexdigest(),
                "trigger_phrases": trigger_phrases,
                "trigger_roots": trigger_roots,
                "protected_roots": protected,
                "prompt": format_revision_prompt(source_sentence, trigger_phrases),
            }
        )
    return tasks


def quick_metrics(per_image):
    rows = list(per_image.values())
    chair_s = sum(row["CHAIRs"] for row in rows) / max(1, len(rows))
    hallucinated = sum(row["hallucinated_mentions"] for row in rows)
    generated = sum(row["generated_mentions"] for row in rows)
    length = sum(row["length"] for row in rows) / max(1, len(rows))
    return {
        "CHAIRs": 100.0 * chair_s,
        "CHAIRi": 100.0 * hallucinated / max(1, generated),
        "Len": length,
        "hallucinated_mentions": hallucinated,
        "generated_mentions": generated,
        "unique_correct_objects": sum(
            len(row["unique_correct"]) for row in rows
        ),
        "unique_hallucinated_objects": sum(
            len(row["unique_hallucinated"]) for row in rows
        ),
        "exact_cycle_images": sum(
            1 for row in rows if row["exact_cycle"]
        ),
    }


def describe_caption(dataset, image_id, text):
    gt_objects, node_words = dataset.eval_hal_words(image_id, text)
    correct = [word for word in node_words if word in gt_objects]
    hallucinated = [word for word in node_words if word not in gt_objects]
    from qwen_vera import exact_cycle

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
    parser.add_argument("--output_dir", default="mgr_ceiling_results")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026071701)
    args = parser.parse_args()

    ids = [int(value) for value in json.load(open(args.ids_file))]
    captions = load_jsonl(args.captions)
    extractions = json.load(open(args.extractions))
    if [int(row["question_id"]) for row in captions] != ids:
        raise ValueError("caption order does not match the frozen dev id list")
    if [int(row["image_id"]) for row in extractions] != ids:
        raise ValueError("extraction order does not match the frozen dev id list")

    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    import contextlib

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        evaluator = dataset.evaluator
    mscoco_objects = set(evaluator.mscoco_objects)

    oracle_records = []
    operator_a_rows = []
    operator_b_rows = []
    operator_c_tasks = []
    per_image = {"vanilla": {}, "operatorA": {}, "operatorB": {}}
    operator_infos = {"operatorA": {}, "operatorB": {}}

    for caption, extraction in zip(captions, extractions):
        image_id = int(caption["question_id"])
        text = caption["text"]
        if hashlib.sha256(text.encode()).hexdigest() != extraction["text_sha256"]:
            raise ValueError(f"source caption changed for {image_id}")
        gt_objects = evaluator.imid_to_objects[image_id]
        _, node_words, _, _ = evaluator.caption_to_words(text)
        hallucinated_nodes = sorted({n for n in node_words if n not in gt_objects})

        triggered_roots = set()
        root_nodes = {}
        for candidate in extraction["candidates"]:
            root = candidate["root_lemma"]
            node = (
                evaluator.inverse_synonym_dict[root]
                if root in mscoco_objects
                else None
            )
            root_nodes[root] = node
            if node is not None and node in hallucinated_nodes:
                triggered_roots.add(root)
        captured = {
            root_nodes[root]
            for root in triggered_roots
            if root_nodes[root] is not None
        }

        revised_a, info_a = apply_operator_a(text, extraction, triggered_roots)
        revised_b, info_b = apply_operator_b(text, extraction, triggered_roots)
        tasks = build_operator_c_tasks(image_id, text, extraction, triggered_roots)

        oracle_records.append(
            {
                "image_id": image_id,
                "hallucinated_nodes": hallucinated_nodes,
                "root_nodes": root_nodes,
                "triggered_roots": sorted(triggered_roots),
                "captured_hallucinated_nodes": sorted(captured),
                "capture_rate": len(captured) / max(1, len(hallucinated_nodes)),
                "operatorA": info_a,
                "operatorB": info_b,
            }
        )
        operator_a_rows.append(
            {"question_id": image_id, "text": revised_a, "method": "oracleA"}
        )
        operator_b_rows.append(
            {"question_id": image_id, "text": revised_b, "method": "oracleB"}
        )
        operator_c_tasks.extend(tasks)
        per_image["vanilla"][str(image_id)] = describe_caption(dataset, image_id, text)
        per_image["operatorA"][str(image_id)] = describe_caption(
            dataset, image_id, revised_a
        )
        per_image["operatorB"][str(image_id)] = describe_caption(
            dataset, image_id, revised_b
        )
        operator_infos["operatorA"][str(image_id)] = info_a
        operator_infos["operatorB"][str(image_id)] = info_b

    output_dir = Path(args.output_dir)
    atomic_json_new(output_dir / "oracle_triggers.json", oracle_records)
    atomic_jsonl_new(output_dir / "operatorA.jsonl", operator_a_rows)
    atomic_jsonl_new(output_dir / "operatorB.jsonl", operator_b_rows)
    atomic_json_new(output_dir / "operatorC_tasks.json", operator_c_tasks)

    metrics = {
        method: quick_metrics(per_image[method])
        for method in ("vanilla", "operatorA", "operatorB")
    }
    hallucinated_images = [
        row for row in oracle_records if row["hallucinated_nodes"]
    ]
    summary = {
        "posthoc_explanation_only": True,
        "must_not_tune_current_dev": True,
        "inputs": {
            "ids_file": args.ids_file,
            "ids_sha256": sha256_file(args.ids_file),
            "captions": args.captions,
            "captions_sha256": sha256_file(args.captions),
            "extractions": args.extractions,
            "extractions_sha256": sha256_file(args.extractions),
            "synonyms_sha256": sha256_file("eval/synonyms.txt"),
        },
        "images": len(ids),
        "images_with_chair_hallucination": len(hallucinated_images),
        "hallucinated_node_word_instances": sum(
            len(row["hallucinated_nodes"]) for row in hallucinated_images
        ),
        "oracle_trigger_tasks": len(operator_c_tasks),
        "oracle_trigger_images": len({row["image_id"] for row in operator_c_tasks}),
        "root_capture_rate_mean": sum(
            row["capture_rate"] for row in hallucinated_images
        )
        / max(1, len(hallucinated_images)),
        "operatorA_fallback_images": sum(
            1 for info in operator_infos["operatorA"].values() if info["fallback_used"]
        ),
        "operatorB_fallback_images": sum(
            1 for info in operator_infos["operatorB"].values() if info["fallback_used"]
        ),
        "metrics": metrics,
        "paired_bootstrap_vs_vanilla": {
            method: bootstrap_delta(
                per_image["vanilla"], per_image[method], args.bootstrap,
                args.bootstrap_seed,
            )
            for method in ("operatorA", "operatorB")
        },
        "per_image": {
            method: {
                key: {
                    "CHAIRs": row["CHAIRs"],
                    "hallucinated_mentions": row["hallucinated_mentions"],
                    "generated_mentions": row["generated_mentions"],
                    "length": row["length"],
                    "unique_correct": sorted(row["unique_correct"]),
                }
                for key, row in per_image[method].items()
            }
            for method in ("vanilla", "operatorA", "operatorB")
        },
        "operator_infos": operator_infos,
        "operatorC_pending": True,
    }
    atomic_json_new(output_dir / "ceiling_analysis.json", summary)
    printable = {key: summary[key] for key in summary if key not in {"per_image", "operator_infos"}}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
