import argparse
import glob
import hashlib
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create disjoint CARE CHAIR splits after excluding every discoverable historical image ID."
    )
    parser.add_argument(
        "--annotations",
        default="data/coco/annotations/captions_val2014.json",
    )
    parser.add_argument("--output_dir", default="experiment")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--dev_size", type=int, default=100)
    parser.add_argument("--validation_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=500)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_ids(value, output):
    if isinstance(value, dict):
        for key in ("question_id", "image_id"):
            candidate = value.get(key)
            if isinstance(candidate, int):
                output.add(candidate)
        for key in ("sentences", "imgToEval", "annotations", "images"):
            if key in value:
                collect_ids(value[key], output)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, int):
                output.add(item)
            else:
                collect_ids(item, output)


def ids_from_json(path):
    found = set()
    try:
        with open(path) as source:
            value = json.load(source)
        collect_ids(value, found)
        return found
    except json.JSONDecodeError:
        pass

    with open(path) as source:
        for line in source:
            if not line.strip():
                continue
            collect_ids(json.loads(line), found)
    return found


def discover_history_files():
    paths = set(glob.glob("experiment/*chair_ids.json"))
    for pattern in ("results/**/*chair*.json", "*results*/**/*chair*.json"):
        paths.update(glob.glob(pattern, recursive=True))
    return sorted(path for path in paths if os.path.isfile(path))


def write_json(path, value):
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite {path}")
    with open(path, "w") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def main():
    args = parse_args()
    requested = args.dev_size + args.validation_size + args.test_size
    if min(args.dev_size, args.validation_size, args.test_size) <= 0:
        raise ValueError("All split sizes must be positive")

    with open(args.annotations) as source:
        annotations = json.load(source)
    all_ids = sorted({int(image["id"]) for image in annotations["images"]})

    history_files = discover_history_files()
    history_by_file = {}
    used_ids = set()
    for path in history_files:
        ids = ids_from_json(path)
        if ids:
            history_by_file[path] = {
                "count": len(ids),
                "sha256": sha256(path),
            }
            used_ids.update(ids)

    available = sorted(set(all_ids) - used_ids)
    if len(available) < requested:
        raise ValueError(f"Need {requested} unused IDs, found {len(available)}")
    rng = random.Random(args.seed)
    rng.shuffle(available)

    dev_end = args.dev_size
    validation_end = dev_end + args.validation_size
    test_end = validation_end + args.test_size
    splits = {
        "dev": available[:dev_end],
        "validation": available[dev_end:validation_end],
        "test": available[validation_end:test_end],
    }
    if len(set().union(*map(set, splits.values()))) != requested:
        raise AssertionError("Generated CARE splits overlap")
    if set().union(*map(set, splits.values())) & used_ids:
        raise AssertionError("Generated CARE splits overlap historical IDs")

    os.makedirs(args.output_dir, exist_ok=True)
    output_paths = {
        name: os.path.join(args.output_dir, f"care_{name}_chair_ids.json")
        for name in splits
    }
    manifest_path = os.path.join(args.output_dir, "care_chair_splits_manifest.json")
    for path in [*output_paths.values(), manifest_path]:
        if os.path.exists(path):
            raise FileExistsError(f"Refusing to overwrite {path}")

    for name, ids in splits.items():
        write_json(output_paths[name], ids)

    manifest = {
        "annotations": args.annotations,
        "annotations_sha256": sha256(args.annotations),
        "seed": args.seed,
        "coco_val_images": len(all_ids),
        "history_files_scanned": len(history_files),
        "history_files_with_ids": history_by_file,
        "historical_unique_ids": len(used_ids),
        "available_unused_ids": len(available),
        "splits": {
            name: {
                "path": path,
                "count": len(splits[name]),
                "sha256": sha256(path),
            }
            for name, path in output_paths.items()
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps({
        "manifest": manifest_path,
        "historical_unique_ids": len(used_ids),
        "available_unused_ids": len(available),
        "splits": {name: len(ids) for name, ids in splits.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
