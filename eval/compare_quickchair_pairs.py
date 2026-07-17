import argparse
import json
import math
import random
import statistics


def load(path):
    with open(path, "r") as handle:
        return json.load(handle)


def aggregate(rows):
    hallucinated = sum(row["hallucinated_mentions"] for row in rows)
    generated = sum(row["generated_mentions"] for row in rows)
    return {
        "samples": len(rows),
        "CHAIRs": 100 * sum(row["CHAIRs"] for row in rows) / len(rows),
        "CHAIRi": 100 * hallucinated / max(1, generated),
        "Len": sum(row["length"] for row in rows) / len(rows),
        "hallucinated_mentions": hallucinated,
        "generated_mentions": generated,
    }


def interval(values):
    values.sort()
    return [
        values[int(0.025 * len(values))],
        values[int(0.975 * len(values)) - 1],
    ]


def mcnemar_exact(new_errors, fixes):
    discordant = new_errors + fixes
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(new_errors, fixes) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def main():
    parser = argparse.ArgumentParser(
        description="Pool multiple disjoint baseline/candidate quick-CHAIR pairs."
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("BASELINE", "CANDIDATE"),
        required=True,
    )
    parser.add_argument("--bootstrap", default=50000, type=int)
    parser.add_argument("--seed", default=20260717, type=int)
    args = parser.parse_args()

    pairs = []
    seen_ids = set()
    set_summaries = []
    total_transitions = {"00": 0, "01": 0, "10": 0, "11": 0}
    for set_index, (baseline_path, candidate_path) in enumerate(args.pair):
        baseline = load(baseline_path)
        candidate = load(candidate_path)
        baseline_rows = baseline["per_image"]
        candidate_rows = candidate["per_image"]
        if set(baseline_rows) != set(candidate_rows):
            raise SystemExit(f"Image IDs differ in pair {set_index}")
        overlap = seen_ids & set(baseline_rows)
        if overlap:
            raise SystemExit(f"Pairs are not disjoint; first overlaps: {sorted(overlap)[:10]}")
        seen_ids.update(baseline_rows)

        transitions = {"00": 0, "01": 0, "10": 0, "11": 0}
        for image_id in sorted(baseline_rows):
            baseline_row = baseline_rows[image_id]
            candidate_row = candidate_rows[image_id]
            key = f"{baseline_row['CHAIRs']}{candidate_row['CHAIRs']}"
            transitions[key] += 1
            total_transitions[key] += 1
            pairs.append((baseline_row, candidate_row))
        set_summaries.append(
            {
                "baseline": baseline_path,
                "candidate": candidate_path,
                "samples": len(baseline_rows),
                "transitions": transitions,
            }
        )

    baseline_metrics = aggregate([baseline for baseline, _ in pairs])
    candidate_metrics = aggregate([candidate for _, candidate in pairs])
    rng = random.Random(args.seed)
    chairs_deltas = []
    chairi_deltas = []
    length_deltas = []
    for _ in range(args.bootstrap):
        sample = [rng.choice(pairs) for _ in pairs]
        baseline_sample = [baseline for baseline, _ in sample]
        candidate_sample = [candidate for _, candidate in sample]
        baseline_agg = aggregate(baseline_sample)
        candidate_agg = aggregate(candidate_sample)
        chairs_deltas.append(candidate_agg["CHAIRs"] - baseline_agg["CHAIRs"])
        chairi_deltas.append(candidate_agg["CHAIRi"] - baseline_agg["CHAIRi"])
        length_deltas.append(candidate_agg["Len"] - baseline_agg["Len"])

    result = {
        "sets": set_summaries,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "CHAIRs_delta": statistics.mean(chairs_deltas),
        "CHAIRs_delta_95ci": interval(chairs_deltas),
        "CHAIRi_delta": statistics.mean(chairi_deltas),
        "CHAIRi_delta_95ci": interval(chairi_deltas),
        "Len_delta": statistics.mean(length_deltas),
        "Len_delta_95ci": interval(length_deltas),
        "transitions": total_transitions,
        "mcnemar_exact_p": mcnemar_exact(total_transitions["01"], total_transitions["10"]),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
