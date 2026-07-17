#!/usr/bin/env python3
"""Merge Qwen CHAIR chunks and materialize open-description RR=SC outputs."""

import argparse
import json
import os
from pathlib import Path


METHODS = ("vanilla", "vhr", "scvhr")


def atomic_jsonl(path, rows):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", required=True)
    parser.add_argument("--input_dir", default="qwen_dev_results")
    parser.add_argument("--num_chunks", type=int, required=True)
    args = parser.parse_args()

    ids = [int(value) for value in json.load(open(args.ids_file))]
    expected = set(ids)
    input_dir = Path(args.input_dir)
    merged = {}
    for method in METHODS:
        by_id = {}
        for chunk in range(args.num_chunks):
            path = input_dir / f"chunk{chunk:02d}-{method}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            for line in path.open():
                row = json.loads(line)
                image_id = int(row["question_id"])
                if image_id in by_id:
                    raise RuntimeError(f"duplicate {method} image id {image_id}")
                by_id[image_id] = row
        if set(by_id) != expected:
            missing = sorted(expected - set(by_id))
            extra = sorted(set(by_id) - expected)
            raise RuntimeError(f"{method} mismatch missing={missing[:10]} extra={extra[:10]}")
        rows = [by_id[image_id] for image_id in ids]
        atomic_jsonl(input_dir / f"{method}.json", rows)
        merged[method] = rows

    rr_rows = []
    for row in merged["scvhr"]:
        rr = dict(row)
        rr["method"] = "rrscvhr"
        rr["response_route"] = "scvhr"
        rr_rows.append(rr)
    atomic_jsonl(input_dir / "rrscvhr.json", rr_rows)
    print(json.dumps({method: len(rows) for method, rows in merged.items()} | {"rrscvhr": len(rr_rows)}))


if __name__ == "__main__":
    main()
