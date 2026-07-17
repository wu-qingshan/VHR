#!/usr/bin/env python3
"""Create a deterministic manifest for the locally downloaded Qwen snapshot."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-vl-7b-instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", default="experiment/qwen25_vl_7b_model_manifest.json")
    args = parser.parse_args()

    root = Path(args.model_path).resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not files:
        raise RuntimeError(f"no model files found under {root}")
    incomplete = list(root.rglob("*.incomplete"))
    if incomplete:
        raise RuntimeError(f"snapshot still has {len(incomplete)} incomplete downloads")

    manifest = {
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "revision": args.revision,
        "local_directory_name": root.name,
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "files": files,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
