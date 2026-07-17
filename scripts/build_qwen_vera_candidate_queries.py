#!/usr/bin/env python3
"""Extract generic noun phrases from VERA base/VHR revision candidates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_sear import extract_caption
from qwen_vera import sanitize_extraction


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def atomic_json_new(path, value, indent=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=indent, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--queries_output", required=True)
    parser.add_argument("--extractions_output", required=True)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import spacy

    nlp = spacy.load(args.spacy_model)
    if str(nlp.meta.get("version")) != "3.8.0":
        raise RuntimeError("VERA requires en_core_web_sm 3.8.0")
    tasks = json.load(open(args.tasks))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    revisions = load_jsonl(args.revisions)
    if [row["revision_id"] for row in revisions] != [row["revision_id"] for row in tasks]:
        raise ValueError("revision rows do not match frozen task order")

    queries = []
    extraction_rows = []
    for task, revision in zip(tasks, revisions):
        modes = {}
        for mode in ("base", "vhr"):
            text = revision["candidates"][mode]["text"]
            started = time.perf_counter()
            extraction = sanitize_extraction(extract_caption(text, nlp))
            extraction_elapsed = time.perf_counter() - started
            for candidate_index, candidate in enumerate(extraction["candidates"]):
                query_id = (
                    f"{task['revision_id']}:{mode}:{candidate_index:03d}:"
                    f"{candidate['root_lemma']}"
                )
                candidate["query_id"] = query_id
                queries.append(
                    {
                        "query_id": query_id,
                        "image_id": int(task["image_id"]),
                        "object": candidate["phrase"],
                        "revision_id": task["revision_id"],
                        "mode": mode,
                        "root_lemma": candidate["root_lemma"],
                        "candidate_index": candidate_index,
                    }
                )
            modes[mode] = {
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "extraction_elapsed_seconds": extraction_elapsed,
                **extraction,
            }
        extraction_rows.append(
            {
                "revision_id": task["revision_id"],
                "image_id": int(task["image_id"]),
                "modes": modes,
            }
        )
    atomic_json_new(args.queries_output, queries, indent=2)
    atomic_json_new(args.extractions_output, extraction_rows)
    print(
        json.dumps(
            {
                "revisions": len(revisions),
                "queries": len(queries),
                "queries_output": args.queries_output,
                "extractions_output": args.extractions_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
