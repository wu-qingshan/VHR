#!/usr/bin/env python3
"""Build the stage-1 mention inventory and query lists (offline, CHAIR-free)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_vera import sanitize_extraction


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def atomic_json_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slugify(key):
    return "_".join(key.split())[:40]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="mgr_verifier_results")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    sets = {
        "old": {
            "captions": "qwen_dev_results/vanilla.json",
            "ids": "experiment/qwen_dev_chair_ids.json",
            "extractions": "qwen_sear_dev_results/extractions.json",
            "verifier": "qwen_sear_dev_results/verifier.json",
        },
        "vera": {
            "captions": "qwen_vera_dev_results/vanilla.json",
            "ids": "experiment/qwen_vera_dev_chair_ids.json",
            "extractions": "qwen_vera_dev_results/source/extractions.json",
            "verifier": "qwen_vera_dev_results/source/verifier.json",
        },
    }

    inventory = []
    whole_queries = []
    inputs_manifest = {}
    for set_name, spec in sets.items():
        captions = load_jsonl(spec["captions"])
        ids = [int(value) for value in json.load(open(spec["ids"]))]
        if [int(row["question_id"]) for row in captions] != ids:
            raise ValueError(f"{set_name}: caption order does not match id list")
        extractions = {
            int(row["image_id"]): row
            for row in json.load(open(spec["extractions"]))
        }
        if [int(row["image_id"]) for row in json.load(open(spec["extractions"]))] != ids:
            raise ValueError(f"{set_name}: extraction order does not match id list")
        materialized = {}
        for row in load_jsonl(spec["verifier"]):
            materialized[(int(row["image_id"]), row["object"].strip().lower())] = {
                "query_id": row["query_id"],
                "score": float(row["score"]),
            }
        selflist_path = output_dir / f"selflist-{set_name}.jsonl"
        selflist = {}
        for row in load_jsonl(selflist_path):
            selflist[int(row["question_id"])] = row
        if sorted(selflist) != ids:
            raise ValueError(f"{set_name}: selflist rows do not cover the id list")
        inputs_manifest[set_name] = {
            key: sha256_file(value) for key, value in spec.items()
        } | {"selflist": sha256_file(selflist_path)}

        for image_id in ids:
            caption = next(
                row["text"] for row in captions if int(row["question_id"]) == image_id
            )
            record = selflist[image_id]
            if (
                hashlib.sha256(caption.encode()).hexdigest()
                != record["caption_sha256"]
            ):
                raise ValueError(f"{set_name}/{image_id}: caption changed under selflist")
            selflist_mentions = record["mentions"]
            extraction = sanitize_extraction(extractions[image_id])
            spacy_candidates = []
            for candidate in extraction["candidates"]:
                spacy_candidates.append(
                    {
                        "root_lemma": candidate["root_lemma"],
                        "phrase": candidate["phrase"],
                        "query_id": candidate["query_id"],
                        "occurrences": candidate["occurrences"],
                        "overlaps_selflist": spans_overlap(
                            candidate["occurrences"], selflist_mentions
                        ),
                    }
                )
            inventory.append(
                {
                    "set": set_name,
                    "image_id": image_id,
                    "caption_sha256": record["caption_sha256"],
                    "selflist_mentions": selflist_mentions,
                    "selflist_unaligned_items": record["unaligned_items"],
                    "selflist_parse_stats": record["parse_stats"],
                    "caption_selflist_empty": len(selflist_mentions) == 0,
                    "spacy_candidates": spacy_candidates,
                }
            )
            sequence = 0
            for mention in selflist_mentions:
                sequence += 1
                add_query(
                    whole_queries,
                    set_name,
                    image_id,
                    mention["mention"],
                    "selflist",
                    mention["occurrences"],
                    materialized,
                    sequence,
                )
            for candidate in spacy_candidates:
                if candidate["overlaps_selflist"]:
                    continue
                sequence += 1
                add_query(
                    whole_queries,
                    set_name,
                    image_id,
                    candidate["phrase"],
                    "spacy",
                    candidate["occurrences"],
                    materialized,
                    sequence,
                )

    atomic_json_new(output_dir / "inventory.json", inventory)
    atomic_json_new(
        output_dir / "whole_queries.json",
        {
            "inputs": inputs_manifest,
            "queries": whole_queries,
        },
    )
    region_tasks = [
        {key: row[key] for key in ("query_id", "set", "image_id", "object", "source")}
        for row in whole_queries
    ]
    atomic_json_new(output_dir / "region_tasks.json", region_tasks)
    print(
        json.dumps(
            {
                "images": len(inventory),
                "selflist_mentions": sum(
                    len(row["selflist_mentions"]) for row in inventory
                ),
                "spacy_nonoverlap": sum(
                    1
                    for row in inventory
                    for c in row["spacy_candidates"]
                    if not c["overlaps_selflist"]
                ),
                "queries": len(whole_queries),
                "reused_whole": sum(
                    1 for row in whole_queries if row["reused_query_id"] is not None
                ),
            },
            indent=2,
        )
    )


def spans_overlap(occurrences, selflist_mentions):
    spans = [
        (int(item["start_char"]), int(item["end_char"]))
        for mention in selflist_mentions
        for item in mention["occurrences"]
    ]
    for occurrence in occurrences:
        start = int(occurrence["start_char"])
        end = int(occurrence["end_char"])
        if any(start < end2 and start2 < end for start2, end2 in spans):
            return True
    return False


def add_query(
    rows, set_name, image_id, object_phrase, source, occurrences, materialized, sequence
):
    key = object_phrase.strip().lower()
    reused = materialized.get((image_id, key))
    rows.append(
        {
            "query_id": f"{set_name}:{image_id}:{sequence:03d}:{slugify(key)}",
            "set": set_name,
            "image_id": image_id,
            "object": object_phrase,
            "object_key": key,
            "source": source,
            "occurrences": occurrences,
            "reused_query_id": reused["query_id"] if reused else None,
            "reused_score": reused["score"] if reused else None,
        }
    )


if __name__ == "__main__":
    main()
