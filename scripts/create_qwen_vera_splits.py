#!/usr/bin/env python3
"""Create fresh VERA splits excluding all historical and reserved COCO IDs."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


ID_FIELDS = {"question_id", "image_id", "coco_id"}


def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_structured_ids(value, valid_ids, output):
    if isinstance(value, dict):
        per_image = value.get("per_image")
        if isinstance(per_image, dict):
            for key in per_image:
                candidate = integer(key)
                if candidate in valid_ids:
                    output.add(candidate)
        for key, child in value.items():
            if key in ID_FIELDS:
                candidate = integer(child)
                if candidate in valid_ids:
                    output.add(candidate)
            if isinstance(child, (dict, list)):
                collect_structured_ids(child, valid_ids, output)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                collect_structured_ids(child, valid_ids, output)


def ids_from_json(path, valid_ids):
    output = set()
    try:
        with path.open() as handle:
            value = json.load(handle)
        if isinstance(value, list) and all(integer(item) is not None for item in value):
            output.update(integer(item) for item in value if integer(item) in valid_ids)
        else:
            collect_structured_ids(value, valid_ids, output)
        return output
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    with path.open(errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            collect_structured_ids(value, valid_ids, output)
    return output


def digest_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path, value):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="val2014")
    parser.add_argument("--seed", type=int, default=2026071701)
    parser.add_argument("--dev_size", type=int, default=300)
    parser.add_argument("--validation_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--extra_exclude", type=int, nargs="*", default=[42, 136])
    parser.add_argument("--output_dir", default="experiment")
    args = parser.parse_args()

    valid_ids = {
        int(path.stem.rsplit("_", 1)[-1])
        for path in Path(args.image_dir).glob("COCO_val2014_*.jpg")
    }
    if len(valid_ids) != 40504:
        raise RuntimeError(f"expected 40504 COCO val2014 images, found {len(valid_ids)}")

    excluded = set()
    source_counts = {}
    id_files = sorted(Path("experiment").glob("**/*ids.json"))
    for path in id_files:
        found = ids_from_json(path, valid_ids)
        if found:
            source_counts[str(path)] = len(found)
            excluded.update(found)

    result_files = []
    for directory in sorted(Path(".").glob("*_results")):
        result_files.extend(sorted(directory.rglob("*.json")))
    for path in result_files:
        found = ids_from_json(path, valid_ids)
        if found:
            source_counts[str(path)] = len(found)
            excluded.update(found)

    explicit = {value for value in args.extra_exclude if value in valid_ids}
    excluded.update(explicit)
    source_counts["explicit_qwen_smoke_ids"] = len(explicit)
    candidates = sorted(valid_ids - excluded)
    needed = args.dev_size + args.validation_size + args.test_size
    if len(candidates) < needed:
        raise RuntimeError(f"only {len(candidates)} unused images remain; need {needed}")
    random.Random(args.seed).shuffle(candidates)
    dev = candidates[: args.dev_size]
    validation = candidates[args.dev_size : args.dev_size + args.validation_size]
    test = candidates[
        args.dev_size + args.validation_size : args.dev_size + args.validation_size + args.test_size
    ]
    assert not (set(dev) & set(validation))
    assert not (set(dev) & set(test))
    assert not (set(validation) & set(test))
    assert not ((set(dev) | set(validation) | set(test)) & excluded)

    output_dir = Path(args.output_dir)
    paths = {
        "dev": output_dir / "qwen_vera_dev_chair_ids.json",
        "validation": output_dir / "qwen_vera_validation_chair_ids.json",
        "test": output_dir / "qwen_vera_test_chair_ids.json",
        "manifest": output_dir / "qwen_vera_split_manifest.json",
    }
    manifest = {
        "seed": args.seed,
        "source_pool": "COCO val2014",
        "source_pool_size": len(valid_ids),
        "excluded_count": len(excluded),
        "candidate_count": len(candidates),
        "explicit_qwen_smoke_ids": sorted(explicit),
        "excluded_ids": sorted(excluded),
        "source_file_counts": source_counts,
        "scanned_id_files": len(id_files),
        "scanned_result_json_files": len(result_files),
        "splits": {
            "dev": {"size": len(dev), "sha256": digest_json(dev)},
            "validation": {"size": len(validation), "sha256": digest_json(validation)},
            "test": {"size": len(test), "sha256": digest_json(test)},
        },
        "pairwise_disjoint": True,
        "disjoint_from_exclusions": True,
    }
    atomic_json(paths["dev"], dev)
    atomic_json(paths["validation"], validation)
    atomic_json(paths["test"], test)
    atomic_json(paths["manifest"], manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "excluded_ids"}, indent=2))


if __name__ == "__main__":
    main()
