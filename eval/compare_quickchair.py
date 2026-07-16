import argparse
import glob
import json
import os
import random
import re
import statistics


def seed_from_path(path):
    match = re.search(r"chair_(\d+)-", os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse CHAIR seed from {path}")
    return int(match.group(1))


def load_group(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched: {pattern}")
    return {seed_from_path(path): json.load(open(path)) for path in files}


def pooled_metrics(group):
    per_image = []
    hall_mentions = generated_mentions = total_len = 0
    for seed, data in group.items():
        rows = list(data["per_image"].values())
        per_image.extend((seed, key, row) for key, row in data["per_image"].items())
        hall_mentions += sum(row["hallucinated_mentions"] for row in rows)
        generated_mentions += sum(row["generated_mentions"] for row in rows)
        total_len += sum(row["length"] for row in rows)
    return {
        "CHAIRs": 100 * sum(row["CHAIRs"] for _, _, row in per_image) / len(per_image),
        "CHAIRi": 100 * hall_mentions / generated_mentions,
        "Len": total_len / len(per_image),
        "num_samples": len(per_image),
    }


def main():
    parser = argparse.ArgumentParser(description="Paired comparison of quick CHAIR result groups.")
    parser.add_argument("--baseline", required=True, help="glob for baseline .quickchair.json files")
    parser.add_argument("--candidate", required=True, help="glob for candidate .quickchair.json files")
    parser.add_argument("--bootstrap", default=20000, type=int)
    parser.add_argument("--seed", default=20260715, type=int)
    args = parser.parse_args()

    baseline = load_group(args.baseline)
    candidate = load_group(args.candidate)
    common_seeds = sorted(baseline.keys() & candidate.keys())
    if common_seeds != sorted(baseline) or common_seeds != sorted(candidate):
        raise SystemExit("Baseline and candidate seed sets differ")

    pairs = []
    transitions = {}
    for seed in common_seeds:
        a = baseline[seed]["per_image"]
        b = candidate[seed]["per_image"]
        keys = sorted(a.keys() & b.keys())
        if len(keys) != len(a) or len(keys) != len(b):
            raise SystemExit(f"Image ids differ for seed {seed}")
        seed_transitions = {"00": 0, "01": 0, "10": 0, "11": 0}
        for key in keys:
            seed_transitions[f"{a[key]['CHAIRs']}{b[key]['CHAIRs']}"] += 1
            pairs.append((a[key], b[key]))
        transitions[seed] = seed_transitions

    rng = random.Random(args.seed)
    deltas = []
    chairi_deltas = []
    length_deltas = []
    for _ in range(args.bootstrap):
        sample = [rng.choice(pairs) for _ in pairs]
        deltas.append(100 * sum(b["CHAIRs"] - a["CHAIRs"] for a, b in sample) / len(sample))
        base_h = sum(a["hallucinated_mentions"] for a, _ in sample)
        base_g = sum(a["generated_mentions"] for a, _ in sample)
        cand_h = sum(b["hallucinated_mentions"] for _, b in sample)
        cand_g = sum(b["generated_mentions"] for _, b in sample)
        chairi_deltas.append(100 * cand_h / max(1, cand_g) - 100 * base_h / max(1, base_g))
        length_deltas.append(sum(b["length"] - a["length"] for a, b in sample) / len(sample))
    deltas.sort()
    chairi_deltas.sort()
    length_deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1]

    def interval(values):
        return [
            values[int(0.025 * len(values))],
            values[int(0.975 * len(values)) - 1],
        ]

    result = {
        "baseline": pooled_metrics(baseline),
        "candidate": pooled_metrics(candidate),
        "CHAIRs_delta": statistics.mean(deltas),
        "CHAIRs_delta_95ci": [lo, hi],
        "probability_delta_below_zero": sum(delta < 0 for delta in deltas) / len(deltas),
        "CHAIRi_delta": statistics.mean(chairi_deltas),
        "CHAIRi_delta_95ci": interval(chairi_deltas),
        "Len_delta": statistics.mean(length_deltas),
        "Len_delta_95ci": interval(length_deltas),
        "transitions_by_seed": transitions,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
