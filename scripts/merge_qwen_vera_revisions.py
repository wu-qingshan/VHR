#!/usr/bin/env python3
"""Strictly merge VERA revision chunks in frozen task order."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--num_chunks", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = json.load(open(args.tasks))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    expected = [row["revision_id"] for row in tasks]
    by_id = {}
    for chunk in range(args.num_chunks):
        path = Path(args.input_dir) / f"chunk{chunk:02d}-revisions.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.open():
            row = json.loads(line)
            revision_id = row["revision_id"]
            if revision_id in by_id:
                raise RuntimeError(f"duplicate revision {revision_id}")
            by_id[revision_id] = row
    if set(by_id) != set(expected):
        raise RuntimeError(
            f"revision mismatch missing={len(set(expected) - set(by_id))} "
            f"extra={len(set(by_id) - set(expected))}"
        )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for revision_id in expected:
            handle.write(json.dumps(by_id[revision_id], ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"revisions": len(expected), "output": args.output}))


if __name__ == "__main__":
    main()
