#!/usr/bin/env python3
"""Strictly merge one Qwen method from GPU chunks in frozen ID order."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--num_chunks", type=int, required=True)
    args = parser.parse_args()

    ids = [int(value) for value in json.load(open(args.ids_file))]
    expected = set(ids)
    root = Path(args.input_dir)
    by_id = {}
    for chunk in range(args.num_chunks):
        path = root / f"chunk{chunk:02d}-{args.method}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.open():
            row = json.loads(line)
            image_id = int(row["question_id"])
            if image_id in by_id:
                raise RuntimeError(f"duplicate image id {image_id}")
            by_id[image_id] = row
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        extra = sorted(set(by_id) - expected)
        raise RuntimeError(f"ID mismatch missing={missing[:10]} extra={extra[:10]}")

    output = root / f"{args.method}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for image_id in ids:
            handle.write(json.dumps(by_id[image_id], ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"method": args.method, "rows": len(by_id), "output": str(output)}))


if __name__ == "__main__":
    main()
