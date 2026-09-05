#!/usr/bin/env python3
"""Analyze the E0.3 grounding smoke: validity, GT IoU, latency, determinism."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench
from pycocotools.coco import COCO
from PIL import Image, ImageDraw, ImageFont


BOX_PATTERN = re.compile(
    r"<box>\s*\[\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\]\s*</box>"
)
BBOX_2D_PATTERN = re.compile(
    r"[\"']bbox_2d[\"']\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
POINT_ONLY_PATTERN = re.compile(r"point_2d|<points")


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


def parse_first_box(raw_text):
    """Prereg v2: accept <box>[[…]]></box> or the first "bbox_2d": [x1,y1,x2,y2]."""

    match = BOX_PATTERN.search(raw_text) or BBOX_2D_PATTERN.search(raw_text)
    if not match:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in match.groups())
    except ValueError:
        return None
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return [x1, y1, x2, y2]


def iou(box, gt_box):
    x1 = max(box[0], gt_box[0])
    y1 = max(box[1], gt_box[1])
    x2 = min(box[2], gt_box[2])
    y2 = min(box[3], gt_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return inter / (area_a + area_b - inter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", default="experiment/qwen_vera_dev_chair_ids.json")
    parser.add_argument("--tasks", default="qwen_vera_dev_results/tasks.json")
    parser.add_argument(
        "--grounding", default="mgr_ceiling_results/e03_grounding.jsonl"
    )
    parser.add_argument(
        "--repeat", default="mgr_ceiling_results/e03_grounding_repeat.jsonl"
    )
    parser.add_argument("--output", default="mgr_ceiling_results/e03_analysis.json")
    parser.add_argument("--overlay_dir", default="mgr_ceiling_results/e03_overlays")
    parser.add_argument("--overlay_seed", type=int, default=2026090402)
    parser.add_argument("--overlay_count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026090401)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    rows = [row for row in load_jsonl(args.grounding) if row["query_type"] == "grounding"]
    repeat_rows = [
        row for row in load_jsonl(args.repeat) if row["query_type"] == "grounding"
    ]
    verifier_checks = {
        row["query_type"]: row
        for row in load_jsonl(args.grounding)
        if row["query_type"].startswith("verifier_check")
    }

    tasks = json.load(open(args.tasks))
    sample = sorted(
        random.Random(args.seed).sample(
            sorted({task["image_id"] for task in tasks}), args.count
        )
    )

    ids = [int(value) for value in json.load(open(args.ids_file))]
    dataset = CHAIRBench(num_samples=len(ids), image_ids=ids)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        evaluator = dataset.evaluator
    mscoco_objects = set(evaluator.mscoco_objects)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        coco = COCO("data/coco/annotations/instances_val2014.json")

    gt_boxes_by_image = {}
    unmapped_categories = set()
    for image_id in sample:
        ann_ids = coco.getAnnIds(imgIds=int(image_id))
        boxes = {}
        for ann in coco.loadAnns(ann_ids):
            category = coco.loadCats(ann["category_id"])[0]["name"]
            if category not in mscoco_objects:
                unmapped_categories.add(category)
                continue
            node = evaluator.inverse_synonym_dict[category]
            boxes.setdefault(node, []).append(
                (ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2],
                 ann["bbox"][1] + ann["bbox"][3])
            )
        gt_boxes_by_image[int(image_id)] = boxes

    per_mention = []
    for row in rows:
        image_id = int(row["image_id"])
        image_path = Path(f"val2014/COCO_val2014_{image_id:012d}.jpg")
        with Image.open(image_path) as image:
            width, height = image.size
        box = parse_first_box(row["raw_text"])
        valid = box is not None
        if valid:
            ungrounded_reason = None
        elif POINT_ONLY_PATTERN.search(row["raw_text"]):
            ungrounded_reason = "point_only"
        else:
            ungrounded_reason = "no_box_output"
        out_of_bounds = valid and (
            box[0] < -1 or box[1] < -1 or box[2] > width + 1 or box[3] > height + 1
        )
        root = row["root_lemma"]
        node = (
            evaluator.inverse_synonym_dict[root] if root in mscoco_objects else None
        )
        gt_objects = evaluator.imid_to_objects[image_id]
        if node is None:
            bucket = "non_chair_root"
            best_iou = None
        elif node in gt_objects:
            bucket = "gt_present"
            candidates = gt_boxes_by_image[image_id].get(node, [])
            best_iou = (
                max(iou(box, gt) for gt in candidates) if valid and candidates else None
            )
        else:
            bucket = "gt_absent"
            best_iou = None
        per_mention.append(
            {
                "mention_id": row["mention_id"],
                "image_id": image_id,
                "phrase": row["phrase"],
                "root_lemma": root,
                "node_word": node,
                "bucket": bucket,
                "raw_text": row["raw_text"],
                "box": box,
                "valid_box": valid,
                "ungrounded_reason": ungrounded_reason,
                "out_of_bounds": out_of_bounds,
                "image_width": width,
                "image_height": height,
                "best_iou": best_iou,
                "iou_at_least_0_5": best_iou is not None and best_iou >= 0.5,
                "elapsed_seconds": row["elapsed_seconds"],
            }
        )

    repeat_by_id = {row["mention_id"]: row for row in repeat_rows}
    determinism_mismatches = [
        row["mention_id"]
        for row in rows
        if row["mention_id"] not in repeat_by_id
        or row["raw_text"] != repeat_by_id[row["mention_id"]]["raw_text"]
        or row["token_ids"] != repeat_by_id[row["mention_id"]]["token_ids"]
    ]

    pre = verifier_checks.get("verifier_check_pre")
    post = verifier_checks.get("verifier_check_post")
    gt_present = [row for row in per_mention if row["bucket"] == "gt_present"]
    valid_gt_present = [row for row in gt_present if row["valid_box"]]

    overlay_dir = Path(args.overlay_dir)
    overlay_index = []
    if args.overlay_count:
        pool = [row for row in valid_gt_present]
        chosen = sorted(
            random.Random(args.overlay_seed).sample(
                pool, min(args.overlay_count, len(pool))
            ),
            key=lambda row: row["mention_id"],
        )
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for row in chosen:
            image_path = Path(f"val2014/COCO_val2014_{row['image_id']:012d}.jpg")
            with Image.open(image_path) as image:
                canvas = image.convert("RGB").copy()
            draw = ImageDraw.Draw(canvas)
            box = row["box"]
            draw.rectangle(
                [box[0], box[1], box[2], box[3]], outline=(255, 0, 0), width=4
            )
            label = f"{row['phrase']} (IoU {row['best_iou']:.2f})" if row["best_iou"] is not None else row["phrase"]
            draw.text((max(2, box[0]), max(2, box[1] - 14)), label, fill=(255, 0, 0))
            overlay_path = overlay_dir / f"{row['mention_id'].replace(':', '_')}.jpg"
            canvas.save(overlay_path)
            overlay_index.append(
                {
                    "mention_id": row["mention_id"],
                    "image_id": row["image_id"],
                    "phrase": row["phrase"],
                    "node_word": row["node_word"],
                    "best_iou": row["best_iou"],
                    "overlay": str(overlay_path),
                }
            )

    latencies = [row["elapsed_seconds"] for row in per_mention]
    report = {
        "images": len(sample),
        "mentions": len(per_mention),
        "valid_box_rate": sum(r["valid_box"] for r in per_mention) / len(per_mention),
        "ungrounded_rate": sum(not r["valid_box"] for r in per_mention) / len(per_mention),
        "ungrounded_reasons": {
            "point_only": sum(
                1 for r in per_mention if r["ungrounded_reason"] == "point_only"
            ),
            "no_box_output": sum(
                1 for r in per_mention if r["ungrounded_reason"] == "no_box_output"
            ),
        },
        "out_of_bounds_count": sum(r["out_of_bounds"] for r in per_mention),
        "buckets": {
            bucket: sum(1 for r in per_mention if r["bucket"] == bucket)
            for bucket in ("gt_present", "gt_absent", "non_chair_root")
        },
        "gt_present_iou_at_least_0_5": {
            "valid_boxes": len(valid_gt_present),
            "hits": sum(r["iou_at_least_0_5"] for r in valid_gt_present),
            "rate": sum(r["iou_at_least_0_5"] for r in valid_gt_present)
            / max(1, len(valid_gt_present)),
        },
        "gt_present_median_iou": sorted(
            r["best_iou"] for r in valid_gt_present
        )[len(valid_gt_present) // 2]
        if valid_gt_present
        else None,
        "latency_seconds": {
            "mean": sum(latencies) / len(latencies),
            "median": sorted(latencies)[len(latencies) // 2],
            "max": max(latencies),
        },
        "determinism": {
            "mismatched_mentions": len(determinism_mismatches),
            "mismatched_ids": determinism_mismatches[:10],
            "passed": not determinism_mismatches,
        },
        "verifier_leak_check": {
            "pre_score": pre["score"] if pre else None,
            "post_score": post["score"] if post else None,
            "passed": bool(pre and post and pre["score"] == post["score"]),
        },
        "unmapped_gt_categories": sorted(unmapped_categories),
        "human_overlay_index": overlay_index,
        "per_mention": per_mention,
    }
    atomic_json_new(args.output, report)
    printable = {k: report[k] for k in report if k != "per_mention"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
