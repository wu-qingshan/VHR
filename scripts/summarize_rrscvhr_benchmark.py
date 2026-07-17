import argparse
import glob
import json
import os
from collections import defaultdict


COMPARE_FIELDS = ("question_id", "text", "gt", "ans_conf", "question")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate and summarize the unified RR-SC-VHR benchmark.")
    parser.add_argument("summary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_relative_range", type=float, default=0.20)
    return parser.parse_args()


def load_output(directory):
    paths = glob.glob(os.path.join(directory, "*", "*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one output JSON under {directory}, found {paths}")
    with open(paths[0]) as source:
        return paths[0], [json.loads(line) for line in source if line.strip()]


def projected(rows):
    return [{field: row.get(field) for field in COMPARE_FIELDS} for row in rows]


def assert_equal(label, left, right):
    if projected(left) != projected(right):
        for index, (left_row, right_row) in enumerate(zip(projected(left), projected(right))):
            if left_row != right_row:
                raise AssertionError(f"{label} differs at row {index}: {left_row} != {right_row}")
        raise AssertionError(f"{label} row counts differ: {len(left)} != {len(right)}")


def mean(values):
    return sum(values) / len(values)


def main():
    args = parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with open(args.summary) as source:
        benchmark = json.load(source)

    grouped = defaultdict(list)
    outputs = {}
    for row in benchmark["results"]:
        key = (row["task"], row["method"])
        grouped[key].append(row)
        path, output_rows = load_output(row["output_dir"])
        outputs[(row["task"], row["method"], row["cycle"])] = output_rows
        row["result_file"] = path

    equivalence = []
    for task in ("chair", "pope"):
        for method in ("vhr", "scvhr", "rrscvhr"):
            assert_equal(
                f"{task}/{method} cycle0 vs cycle1",
                outputs[(task, method, 0)],
                outputs[(task, method, 1)],
            )
            equivalence.append(f"{task}/{method}: cycle0 == cycle1")
    for cycle in (0, 1):
        assert_equal(
            f"chair RR vs SC cycle{cycle}",
            outputs[("chair", "rrscvhr", cycle)],
            outputs[("chair", "scvhr", cycle)],
        )
        assert_equal(
            f"pope RR vs VHR cycle{cycle}",
            outputs[("pope", "rrscvhr", cycle)],
            outputs[("pope", "vhr", cycle)],
        )
        equivalence.extend([
            f"chair cycle{cycle}: RR == SC",
            f"pope cycle{cycle}: RR == VHR",
        ])

    aggregate = {}
    needs_third_cycle = []
    for (task, method), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["cycle"])
        seconds = [row["seconds_per_sample"] for row in rows]
        relative_range = (max(seconds) - min(seconds)) / mean(seconds)
        if relative_range > args.max_relative_range:
            needs_third_cycle.append({"task": task, "method": method, "relative_range": relative_range})
        aggregate.setdefault(task, {})[method] = {
            "cycles": rows,
            "mean_seconds_per_sample": mean(seconds),
            "mean_samples_per_second": mean([row["samples_per_second"] for row in rows]),
            "relative_latency_range": relative_range,
            "mean_peak_allocated_gib": mean([row["peak_allocated_gib"] for row in rows]),
            "mean_incremental_peak_allocated_gib": mean(
                [row["incremental_peak_allocated_gib"] for row in rows]
            ),
        }

    ratios = {}
    for task, methods in aggregate.items():
        latency = {method: values["mean_seconds_per_sample"] for method, values in methods.items()}
        ratios[task] = {
            "sc_over_vhr": latency["scvhr"] / latency["vhr"],
            "rr_over_vhr": latency["rrscvhr"] / latency["vhr"],
            "rr_over_sc": latency["rrscvhr"] / latency["scvhr"],
            "rr_speedup_vs_sc": latency["scvhr"] / latency["rrscvhr"],
        }

    report = {
        "source": args.summary,
        "compare_fields": COMPARE_FIELDS,
        "equivalence_checks": equivalence,
        "aggregate": aggregate,
        "latency_ratios": ratios,
        "third_cycle_threshold": args.max_relative_range,
        "needs_third_cycle": needs_third_cycle,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
