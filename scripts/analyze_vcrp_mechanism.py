import argparse
import contextlib
import glob
import json
import os
import random
import statistics
import sys
from collections import defaultdict

import nltk
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench


PRIMARY_SIGNAL = "opposing_ffn_fraction"
SIGNALS = (
    PRIMARY_SIGNAL,
    "conflict_layer_fraction",
    "visual_ffn_norm_ratio",
    "projection_hidden_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Align no-intervention VCRP token diagnostics with CHAIR object mentions "
            "and evaluate the preregistered mechanism gate."
        )
    )
    parser.add_argument("--inputs", required=True, help="Glob for vcrpdiag JSON files")
    parser.add_argument("--model_path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", default=20000, type=int)
    parser.add_argument("--seed", default=20260717, type=int)
    return parser.parse_args()


def roc_auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def image_bootstrap_median_difference(mentions, iterations, seed):
    by_image = defaultdict(list)
    for mention in mentions:
        by_image[int(mention["image_id"])].append(mention)
    image_ids = sorted(by_image)
    rng = random.Random(seed)
    differences = []
    for _ in range(iterations):
        sampled = [rng.choice(image_ids) for _ in image_ids]
        hallucinated = []
        correct = []
        for image_id in sampled:
            for mention in by_image[image_id]:
                destination = hallucinated if mention["hallucinated"] else correct
                destination.append(mention[PRIMARY_SIGNAL])
        if hallucinated and correct:
            differences.append(
                statistics.median(hallucinated) - statistics.median(correct)
            )
    if not differences:
        return None
    differences.sort()
    lower = differences[int(0.025 * len(differences))]
    upper = differences[min(len(differences) - 1, int(0.975 * len(differences)))]
    return [lower, upper]


def generated_token_end_positions(tokenizer, token_ids):
    positions = []
    previous = ""
    for end in range(1, len(token_ids) + 1):
        current = tokenizer.decode(token_ids[:end], skip_special_tokens=True)
        if not current.startswith(previous):
            raise ValueError("Generated token decoding is not prefix-monotonic")
        positions.append(len(current))
        previous = current
    return previous, positions


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
    ids = [int(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs across VCRP diagnostic inputs")

    treebank = TreebankWordTokenizer()
    # CHAIR captions are single sentences.  ``preserve_line=True`` semantics
    # avoid a version-specific NLTK punkt_tab download while retaining the
    # Treebank tokenization used by word_tokenize after sentence splitting.
    nltk.word_tokenize = lambda text, *unused_args, **unused_kwargs: treebank.tokenize(text)
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    dataset = CHAIRBench(image_ids=ids)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        evaluator = dataset.evaluator
    mentions = []
    alignment_failures = []

    for row in rows:
        image_id = int(row["question_id"])
        token_stats = row["vcrp_stats"]["tokens"]
        token_ids = [int(item["selected_token_id"]) for item in token_stats]
        decoded, token_end_positions = generated_token_end_positions(tokenizer, token_ids)
        if decoded.strip() != row["text"].strip():
            alignment_failures.append([image_id, "decoded_text_mismatch"])
            continue

        text = row["text"]
        word_spans = list(treebank.span_tokenize(text.lower()))
        nltk_words = nltk.word_tokenize(text.lower())
        span_words = [text.lower()[start:end] for start, end in word_spans]
        if len(nltk_words) != len(word_spans):
            alignment_failures.append([image_id, "word_span_count_mismatch"])
            continue
        words, node_words, word_indices, _ = evaluator.caption_to_words(text)
        gt_objects = evaluator.imid_to_objects[image_id]
        for word, node_word, word_index in zip(words, node_words, word_indices):
            if not 0 <= word_index < len(word_spans):
                alignment_failures.append([image_id, "invalid_word_index"])
                continue
            end_word_index = word_index + (1 if " " in word else 0)
            end_word_index = min(end_word_index, len(word_spans) - 1)
            object_end = word_spans[end_word_index][1]
            token_index = next(
                (
                    index
                    for index, token_end in enumerate(token_end_positions)
                    if token_end >= object_end
                ),
                None,
            )
            if token_index is None or token_index >= len(token_stats):
                alignment_failures.append([image_id, "object_token_not_found"])
                continue
            diagnostic = token_stats[token_index]
            mention = {
                "image_id": image_id,
                "word": word,
                "node_word": node_word,
                "raw_word": span_words[word_index],
                "word_index": int(word_index),
                "token_index": int(token_index),
                "relative_position": token_index / max(1, len(token_stats) - 1),
                "hallucinated": node_word not in gt_objects,
            }
            for signal in SIGNALS:
                mention[signal] = float(diagnostic[signal])
            mentions.append(mention)

    hallucinated = [mention for mention in mentions if mention["hallucinated"]]
    correct = [mention for mention in mentions if not mention["hallucinated"]]
    labels = [int(mention["hallucinated"]) for mention in mentions]
    signal_summaries = {}
    for signal in SIGNALS:
        hallucinated_values = [mention[signal] for mention in hallucinated]
        correct_values = [mention[signal] for mention in correct]
        signal_summaries[signal] = {
            "hallucinated_median": statistics.median(hallucinated_values)
            if hallucinated_values
            else None,
            "correct_median": statistics.median(correct_values) if correct_values else None,
            "median_difference": (
                statistics.median(hallucinated_values) - statistics.median(correct_values)
                if hallucinated_values and correct_values
                else None
            ),
            "roc_auc": roc_auc(labels, [mention[signal] for mention in mentions]),
        }

    quartiles = []
    directional_quartiles = 0
    for quartile in range(4):
        lower, upper = quartile / 4, (quartile + 1) / 4
        subset = [
            mention
            for mention in mentions
            if lower <= mention["relative_position"] < upper
            or (quartile == 3 and mention["relative_position"] == 1.0)
        ]
        subset_h = [mention[PRIMARY_SIGNAL] for mention in subset if mention["hallucinated"]]
        subset_c = [mention[PRIMARY_SIGNAL] for mention in subset if not mention["hallucinated"]]
        difference = (
            statistics.median(subset_h) - statistics.median(subset_c)
            if subset_h and subset_c
            else None
        )
        directional_quartiles += int(difference is not None and difference > 0)
        quartiles.append({
            "quartile": quartile + 1,
            "hallucinated_mentions": len(subset_h),
            "correct_mentions": len(subset_c),
            "median_difference": difference,
        })

    primary = signal_summaries[PRIMARY_SIGNAL]
    bootstrap_ci = image_bootstrap_median_difference(
        mentions, args.bootstrap, args.seed
    )
    gates = {
        "at_least_50_hallucinated_mentions": len(hallucinated) >= 50,
        "at_least_100_correct_mentions": len(correct) >= 100,
        "hallucinated_median_higher": (
            primary["median_difference"] is not None and primary["median_difference"] > 0
        ),
        "bootstrap_ci_lower_above_zero": (
            bootstrap_ci is not None and bootstrap_ci[0] > 0
        ),
        "roc_auc_at_least_065": (
            primary["roc_auc"] is not None and primary["roc_auc"] >= 0.65
        ),
        "at_least_three_directional_position_quartiles": directional_quartiles >= 3,
        "no_alignment_failures": not alignment_failures,
    }
    result = {
        "purpose": "VCRP no-intervention mechanism diagnostic; no effect result",
        "inputs": paths,
        "images": len(rows),
        "object_mentions": len(mentions),
        "hallucinated_mentions": len(hallucinated),
        "correct_mentions": len(correct),
        "alignment_failures": alignment_failures,
        "signals": signal_summaries,
        "primary_bootstrap_median_difference_95ci": bootstrap_ci,
        "position_quartiles": quartiles,
        "directional_quartiles": directional_quartiles,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "mentions": mentions,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "x") as destination:
        json.dump(result, destination, indent=2)
    printable = dict(result)
    printable.pop("mentions")
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
