#!/usr/bin/env python3
"""Build generic SEAR phrase queries from frozen Qwen vanilla captions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_sear import extract_caption


def atomic_new(path, writer):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        writer(handle)
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", default="experiment/qwen_dev_chair_ids.json")
    parser.add_argument("--captions", default="qwen_dev_results/vanilla.json")
    parser.add_argument("--output_dir", default="qwen_sear_dev_results")
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()

    import spacy

    nlp = spacy.load(args.spacy_model)
    if str(nlp.meta.get("version")) != "3.8.0":
        raise RuntimeError("SEAR requires en_core_web_sm 3.8.0")
    ids = [int(value) for value in json.load(open(args.ids_file))]
    rows = [json.loads(line) for line in open(args.captions) if line.strip()]
    if [int(row["question_id"]) for row in rows] != ids:
        raise ValueError("captions do not match frozen Qwen dev IDs in exact order")

    queries = []
    extractions = []
    for row in rows:
        image_id = int(row["question_id"])
        started = time.perf_counter()
        extraction = extract_caption(row["text"], nlp)
        extraction_elapsed = time.perf_counter() - started
        for candidate_index, candidate in enumerate(extraction["candidates"]):
            query_id = f"{image_id}:{candidate_index:03d}:{candidate['root_lemma']}"
            candidate["query_id"] = query_id
            queries.append(
                {
                    "query_id": query_id,
                    "image_id": image_id,
                    "object": candidate["phrase"],
                    "root_lemma": candidate["root_lemma"],
                    "candidate_index": candidate_index,
                }
            )
        extractions.append(
            {
                "image_id": image_id,
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "extraction_elapsed_seconds": extraction_elapsed,
                **extraction,
            }
        )

    output_dir = Path(args.output_dir)
    queries_path = output_dir / "queries.json"
    extractions_path = output_dir / "extractions.json"
    manifest_path = output_dir / "query_manifest.json"
    atomic_new(
        queries_path,
        lambda handle: handle.write(json.dumps(queries, indent=2, ensure_ascii=False) + "\n"),
    )
    atomic_new(
        extractions_path,
        lambda handle: handle.write(json.dumps(extractions, ensure_ascii=False) + "\n"),
    )
    lengths = [len(query["object"].split()) for query in queries]
    manifest = {
        "ids_file": args.ids_file,
        "ids_sha256": sha256(args.ids_file),
        "captions": args.captions,
        "captions_sha256": sha256(args.captions),
        "spacy_version": spacy.__version__,
        "spacy_model": args.spacy_model,
        "spacy_model_version": nlp.meta.get("version"),
        "images": len(rows),
        "queries": len(queries),
        "mean_queries_per_image": len(queries) / max(1, len(rows)),
        "mean_query_words": sum(lengths) / max(1, len(lengths)),
        "max_query_words": max(lengths, default=0),
        "mean_extraction_seconds": sum(
            row["extraction_elapsed_seconds"] for row in extractions
        )
        / max(1, len(extractions)),
        "chair_vocabulary_used": False,
    }
    atomic_new(
        manifest_path,
        lambda handle: handle.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
