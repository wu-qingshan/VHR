import argparse
import json
import os


def truncate_consecutive_cycle(text, max_period=16, repeats=3):
    tokens = text.split()
    for end in range(repeats, len(tokens) + 1):
        max_candidate_period = min(max_period, end // repeats)
        for period in range(1, max_candidate_period + 1):
            block = tokens[end - period:end]
            if all(
                tokens[end - (repeat_idx + 1) * period:end - repeat_idx * period] == block
                for repeat_idx in range(1, repeats)
            ):
                cycle_start = end - repeats * period
                return " ".join(tokens[:cycle_start]), {
                    "cycle_start": cycle_start,
                    "period": period,
                    "block": block,
                }
    return text, None


def main():
    parser = argparse.ArgumentParser(description="Truncate only exact consecutive token cycles.")
    parser.add_argument("result_files", nargs="+")
    parser.add_argument("--max_period", default=16, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--suffix", default="cycle3")
    args = parser.parse_args()

    for result_file in args.result_files:
        output_file = result_file.replace(".json", f"-{args.suffix}.json")
        hits = []
        with open(output_file, "w") as output:
            for line in open(result_file):
                row = json.loads(line)
                truncated, info = truncate_consecutive_cycle(
                    row["text"],
                    max_period=args.max_period,
                    repeats=args.repeats,
                )
                if info is not None:
                    hits.append({
                        "question_id": row["question_id"],
                        "original_length": len(row["text"].split()),
                        "truncated_length": len(truncated.split()),
                        **info,
                    })
                    row["text"] = truncated
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({
            "input": result_file,
            "output": output_file,
            "num_truncated": len(hits),
            "hits": hits,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()

