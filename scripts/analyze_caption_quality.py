import argparse
import glob
import json
import math
import os
import re
from collections import Counter


WORD_RE = re.compile(r"\w+(?:['’]\w+)?", flags=re.UNICODE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare lexical diversity, repeated n-grams, and exact cycles in caption files."
    )
    parser.add_argument("--vhr", required=True, help="Glob for the VHR JSONL files.")
    parser.add_argument("--rr", required=True, help="Glob for the RR-SC-VHR JSONL files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_cycle_period", type=int, default=16)
    parser.add_argument("--cycle_repeats", type=int, default=3)
    return parser.parse_args()


def word_tokens(text):
    return WORD_RE.findall(text.lower())


def ngrams(tokens, n):
    return [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]


def exact_cycle(text, max_period, repeats):
    tokens = text.split()
    for end in range(repeats, len(tokens) + 1):
        max_candidate_period = min(max_period, end // repeats)
        for period in range(1, max_candidate_period + 1):
            block = tokens[end - period:end]
            if all(
                tokens[end - (repeat_index + 1) * period:end - repeat_index * period] == block
                for repeat_index in range(1, repeats)
            ):
                return {
                    "cycle_start": end - repeats * period,
                    "period": period,
                    "block": block,
                }
    return None


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def load_rows(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def caption_metrics(row, max_cycle_period, cycle_repeats):
    tokens = word_tokens(row["text"])
    result = {"word_count": len(tokens)}
    for n in (1, 2):
        grams = ngrams(tokens, n)
        result[f"distinct_{n}"] = safe_ratio(len(set(grams)), len(grams))
    for n in (2, 3, 4):
        grams = ngrams(tokens, n)
        repeated = len(grams) - len(set(grams))
        result[f"repeated_{n}gram_ratio"] = safe_ratio(repeated, len(grams))
    result["exact_cycle"] = exact_cycle(row["text"], max_cycle_period, cycle_repeats) is not None
    return result


def analyze_rows(rows, max_cycle_period, cycle_repeats):
    corpus_tokens = []
    per_caption = {
        "word_count": [],
        "distinct_1": [],
        "distinct_2": [],
        "repeated_2gram_ratio": [],
        "repeated_3gram_ratio": [],
        "repeated_4gram_ratio": [],
    }
    cycle_hits = []

    for row in rows:
        tokens = word_tokens(row["text"])
        corpus_tokens.extend(tokens)
        metrics = caption_metrics(row, max_cycle_period, cycle_repeats)
        for name in per_caption:
            per_caption[name].append(metrics[name])

        cycle = exact_cycle(row["text"], max_cycle_period, cycle_repeats)
        if cycle is not None:
            cycle_hits.append({"question_id": row["question_id"], **cycle})

    corpus = {}
    for n in (1, 2):
        grams = ngrams(corpus_tokens, n)
        corpus[f"distinct_{n}"] = safe_ratio(len(set(grams)), len(grams))
    for n in (2, 3, 4):
        grams = ngrams(corpus_tokens, n)
        repeated = len(grams) - len(set(grams))
        corpus[f"repeated_{n}gram_ratio"] = safe_ratio(repeated, len(grams))

    return {
        "captions": len(rows),
        "tokens": len(corpus_tokens),
        "mean_per_caption": {name: mean(values) for name, values in per_caption.items()},
        "corpus": corpus,
        "exact_cycle": {
            "hits": len(cycle_hits),
            "hit_rate": safe_ratio(len(cycle_hits), len(rows)),
            "examples": cycle_hits,
        },
    }


def analyze_group(pattern, max_cycle_period, cycle_repeats):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    per_file = {}
    all_rows = []
    for path in paths:
        rows = load_rows(path)
        all_rows.extend(rows)
        per_file[os.path.basename(path)] = analyze_rows(rows, max_cycle_period, cycle_repeats)
    return {
        "files": paths,
        "per_file": per_file,
        "combined": analyze_rows(all_rows, max_cycle_period, cycle_repeats),
    }


def numeric_delta(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return {
            key: numeric_delta(left[key], right[key])
            for key in left.keys() & right.keys()
            if key not in {"examples"}
        }
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return right - left
    return None


def exact_mcnemar_p(left_only, right_only):
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / 2**discordant
    return min(1.0, 2 * probability)


def paired_comparison(vhr_paths, rr_paths, max_cycle_period, cycle_repeats):
    if len(vhr_paths) != len(rr_paths):
        raise ValueError("VHR and RR file counts differ")
    deltas = {
        "word_count": [],
        "distinct_1": [],
        "distinct_2": [],
        "repeated_2gram_ratio": [],
        "repeated_3gram_ratio": [],
        "repeated_4gram_ratio": [],
    }
    cycle_counts = {"neither": 0, "vhr_only": 0, "rr_only": 0, "both": 0}
    pairs = 0
    for vhr_path, rr_path in zip(vhr_paths, rr_paths):
        vhr_rows = load_rows(vhr_path)
        rr_rows = load_rows(rr_path)
        if len(vhr_rows) != len(rr_rows):
            raise ValueError(f"Row count differs: {vhr_path} vs {rr_path}")
        for vhr_row, rr_row in zip(vhr_rows, rr_rows):
            if vhr_row["question_id"] != rr_row["question_id"]:
                raise ValueError(f"Question ID mismatch: {vhr_path} vs {rr_path}")
            vhr_metrics = caption_metrics(vhr_row, max_cycle_period, cycle_repeats)
            rr_metrics = caption_metrics(rr_row, max_cycle_period, cycle_repeats)
            for name in deltas:
                deltas[name].append(rr_metrics[name] - vhr_metrics[name])
            if vhr_metrics["exact_cycle"] and rr_metrics["exact_cycle"]:
                cycle_counts["both"] += 1
            elif vhr_metrics["exact_cycle"]:
                cycle_counts["vhr_only"] += 1
            elif rr_metrics["exact_cycle"]:
                cycle_counts["rr_only"] += 1
            else:
                cycle_counts["neither"] += 1
            pairs += 1
    cycle_counts["mcnemar_exact_p"] = exact_mcnemar_p(
        cycle_counts["vhr_only"], cycle_counts["rr_only"]
    )
    return {
        "pairs": pairs,
        "mean_delta_rr_minus_vhr": {name: mean(values) for name, values in deltas.items()},
        "exact_cycle_contingency": cycle_counts,
    }


def main():
    args = parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    vhr = analyze_group(args.vhr, args.max_cycle_period, args.cycle_repeats)
    rr = analyze_group(args.rr, args.max_cycle_period, args.cycle_repeats)
    if vhr["combined"]["captions"] != rr["combined"]["captions"]:
        raise ValueError("VHR and RR caption counts differ")

    report = {
        "definitions": {
            "tokenization": "lowercased Unicode word tokens; punctuation excluded",
            "distinct_n": "unique n-grams / all n-grams",
            "repeated_ngram_ratio": "(all n-gram occurrences - unique n-grams) / all n-gram occurrences",
            "exact_cycle": (
                f"case-sensitive whitespace-token block with period <= {args.max_cycle_period} "
                f"appearing consecutively >= {args.cycle_repeats} times"
            ),
        },
        "vhr": vhr,
        "rr_scvhr": rr,
        "delta_rr_minus_vhr": numeric_delta(vhr["combined"], rr["combined"]),
        "paired": paired_comparison(
            vhr["files"], rr["files"], args.max_cycle_period, args.cycle_repeats
        ),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
    print(json.dumps({
        "output": args.output,
        "vhr": vhr["combined"],
        "rr_scvhr": rr["combined"],
        "delta_rr_minus_vhr": report["delta_rr_minus_vhr"],
        "paired": report["paired"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
