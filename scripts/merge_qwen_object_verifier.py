#!/usr/bin/env python3
"""Strictly merge object verifier chunks in frozen query order."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="qwen_object_evidence_results/queries.json")
    parser.add_argument("--input_dir", default="qwen_object_evidence_results")
    parser.add_argument("--num_chunks", type=int, default=6)
    parser.add_argument("--output", default="qwen_object_evidence_results/verifier.json")
    args = parser.parse_args()

    queries = json.load(open(args.queries))
    expected = [row["query_id"] for row in queries]
    by_id = {}
    root = Path(args.input_dir)
    for chunk in range(args.num_chunks):
        path = root / f"chunk{chunk:02d}-verifier.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.open():
            row = json.loads(line)
            if row["query_id"] in by_id:
                raise RuntimeError(f"duplicate query {row['query_id']}")
            by_id[row["query_id"]] = row
    if set(by_id) != set(expected):
        missing = sorted(set(expected) - set(by_id))
        extra = sorted(set(by_id) - set(expected))
        raise RuntimeError(f"query mismatch missing={missing[:10]} extra={extra[:10]}")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for query_id in expected:
            handle.write(json.dumps(by_id[query_id], ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"queries": len(expected), "output": str(output)}))


if __name__ == "__main__":
    main()
