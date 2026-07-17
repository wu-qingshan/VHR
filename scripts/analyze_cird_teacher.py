import argparse
import glob
import json
import os
import random
import statistics
import sys
from collections import defaultdict

import nltk
from nltk.tokenize import TreebankWordTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.chair import CHAIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the preregistered CIRD teacher-forcing feasibility gate."
    )
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", default=20000, type=int)
    parser.add_argument("--seed", default=20260717, type=int)
    return parser.parse_args()


def bootstrap_mean(values, iterations, seed):
    if not values:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [rng.choice(values) for _ in values]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return [
        samples[int(0.025 * len(samples))],
        samples[min(len(samples) - 1, int(0.975 * len(samples)))],
    ]


def token_end_positions(tokenizer, token_ids):
    positions = []
    previous = ""
    for end in range(1, len(token_ids) + 1):
        current = tokenizer.decode(token_ids[:end], skip_special_tokens=True)
        if not current.startswith(previous):
            raise ValueError("Token decoding is not prefix-monotonic")
        positions.append(len(current))
        previous = current
    return positions


def mean_available(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return sum(values) / len(values) if values else None


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.inputs, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No files matched {args.inputs!r}")
    if os.path.exists(args.output):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    rows = []
    for path in paths:
        with open(path) as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    image_ids = [int(row["image_id"]) for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate CIRD teacher images")

    treebank = TreebankWordTokenizer()
    nltk.word_tokenize = lambda text, *unused_args, **unused_kwargs: treebank.tokenize(text)
    chair = CHAIR([], "data/coco/annotations")
    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained("models/llava-1.5-7b-hf").tokenizer
    all_tokens = []
    object_tokens = []
    nonobject_tokens = []
    alignment_failures = []
    per_image_object_delta = []

    for row in rows:
        image_id = int(row["image_id"])
        caption = row["caption"]
        token_ids = [int(token_id) for token_id in row["forced_token_ids"]]
        stats = row["teacher_stats"]["tokens"]
        if len(stats) != len(token_ids):
            alignment_failures.append([image_id, "stat_length_mismatch"])
            continue
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        if decoded.strip() != caption.strip():
            alignment_failures.append([image_id, "caption_roundtrip_mismatch"])
            continue
        positions = token_end_positions(tokenizer, token_ids)
        word_spans = list(treebank.span_tokenize(caption.lower()))
        words, _, word_indices, _ = chair.caption_to_words(caption)
        object_indices = set()
        for word, word_index in zip(words, word_indices):
            end_word_index = min(
                word_index + (1 if " " in word else 0), len(word_spans) - 1
            )
            object_end = word_spans[end_word_index][1]
            token_index = next(
                (index for index, end in enumerate(positions) if end >= object_end),
                None,
            )
            if token_index is None:
                alignment_failures.append([image_id, "object_token_not_found"])
            else:
                object_indices.add(token_index)

        image_object_rows = []
        for index, stat in enumerate(stats):
            enriched = {
                **stat,
                "image_id": image_id,
                "token_index": index,
                "is_object": index in object_indices,
            }
            all_tokens.append(enriched)
            if index in object_indices:
                object_tokens.append(enriched)
                image_object_rows.append(enriched)
            else:
                nonobject_tokens.append(enriched)
        if image_object_rows and all(item["sc_nll"] is not None for item in image_object_rows):
            per_image_object_delta.append(
                sum(item["sc_nll"] - item["base_nll"] for item in image_object_rows)
                / len(image_object_rows)
            )

    unavailable_sc = sum(token["sc_nll"] is None for token in all_tokens)
    unavailable_object_sc = sum(token["sc_nll"] is None for token in object_tokens)
    object_sc = mean_available(object_tokens, "sc_nll")
    object_base = mean_available(object_tokens, "base_nll")
    object_vhr = mean_available(object_tokens, "vhr_nll")
    all_sc = mean_available(all_tokens, "sc_nll")
    all_base = mean_available(all_tokens, "base_nll")
    bootstrap_ci = bootstrap_mean(per_image_object_delta, args.bootstrap, args.seed)
    improved_images = sum(delta < 0 for delta in per_image_object_delta)
    improved_fraction = improved_images / max(1, len(per_image_object_delta))

    gates = {
        "100_images_complete": len(rows) == 100,
        "no_alignment_failures": not alignment_failures,
        "all_sc_logprobs_finite": unavailable_sc == 0,
        "at_least_200_object_tokens": len(object_tokens) >= 200,
        "object_sc_better_than_base": (
            object_sc is not None and object_base is not None and object_sc < object_base
        ),
        "object_bootstrap_ci_upper_below_zero": (
            bootstrap_ci is not None and bootstrap_ci[1] < 0
        ),
        "object_sc_not_worse_than_vhr": (
            object_sc is not None and object_vhr is not None and object_sc <= object_vhr
        ),
        "all_token_nll_regression_at_most_002": (
            all_sc is not None and all_base is not None and all_sc - all_base <= 0.02
        ),
        "object_improved_image_fraction_at_least_060": improved_fraction >= 0.60,
    }
    result = {
        "purpose": "CIRD frozen teacher-forcing feasibility; no training/effect result",
        "inputs": paths,
        "images": len(rows),
        "tokens": len(all_tokens),
        "object_tokens": len(object_tokens),
        "nonobject_tokens": len(nonobject_tokens),
        "sc_unavailable_tokens": unavailable_sc,
        "sc_unavailable_token_fraction": unavailable_sc / max(1, len(all_tokens)),
        "sc_unavailable_object_tokens": unavailable_object_sc,
        "sc_unavailable_object_fraction": unavailable_object_sc
        / max(1, len(object_tokens)),
        "mean_nll": {
            "all": {"base": all_base, "vhr": mean_available(all_tokens, "vhr_nll"), "sc": all_sc},
            "object": {"base": object_base, "vhr": object_vhr, "sc": object_sc},
            "nonobject": {
                "base": mean_available(nonobject_tokens, "base_nll"),
                "vhr": mean_available(nonobject_tokens, "vhr_nll"),
                "sc": mean_available(nonobject_tokens, "sc_nll"),
            },
        },
        "complete_object_images": len(per_image_object_delta),
        "object_sc_minus_base_image_mean_95ci": bootstrap_ci,
        "object_improved_images": improved_images,
        "object_improved_image_fraction": improved_fraction,
        "alignment_failures": alignment_failures,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "x") as destination:
        json.dump(result, destination, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
