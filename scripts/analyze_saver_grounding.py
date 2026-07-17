import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from pycocotools.coco import COCO
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saver import patch_bbox_to_raw_bbox


VARIANTS = ("selected", "all_head", "content", "value")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SAVER diagnostic maps against COCO instance masks. "
            "This is a grounding diagnostic and does not compute CHAIR."
        )
    )
    parser.add_argument("--inputs", required=True, help="Glob for diagnostic JSON files")
    parser.add_argument("--model_path", default="models/llava-1.5-7b-hf")
    parser.add_argument(
        "--instances", default="data/coco/annotations/instances_val2014.json"
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def normalize_category_token(text):
    token = re.sub(r"[^a-z ]", "", text.strip().lower())
    if token.endswith("ies") and len(token) > 3:
        token = token[:-3] + "y"
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 2:
        token = token[:-1]
    return token


def variant_fields(example, variant):
    if variant == "selected":
        return example["bbox"], example["top_indices"]
    return example[f"{variant}_bbox"], example[f"{variant}_top_indices"]


def mcnemar_exact_pvalue(left_only, right_only):
    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    tail = min(int(left_only), int(right_only))
    probability = 2.0 * sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, probability)


def point_from_patch(index, grid_side, image_size):
    y, x = divmod(int(index), int(grid_side))
    mapping = patch_bbox_to_raw_bbox(
        [x, y, x + 1, y + 1], grid_side, image_size, 336, (336, 336)
    )
    x0, y0, x1, y1 = mapping["raw_bbox"]
    return min(image_size[0] - 1, (x0 + x1 - 1) // 2), min(
        image_size[1] - 1, (y0 + y1 - 1) // 2
    )


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.inputs, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No diagnostic files matched {args.inputs!r}")
    if os.path.exists(args.output):
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    coco = COCO(args.instances)
    category_names = {
        int(category["id"]): normalize_category_token(category["name"])
        for category in coco.loadCats(coco.getCatIds())
    }
    category_ids_by_name = defaultdict(list)
    for category_id, name in category_names.items():
        category_ids_by_name[name].append(category_id)

    totals = {
        variant: Counter(
            steps=0,
            any_point_hits=0,
            any_crop_hits=0,
            candidate_steps=0,
            candidate_point_hits=0,
            candidate_crop_hits=0,
            center_candidate_hits=0,
        )
        for variant in VARIANTS
    }
    switch_tokens = Counter()
    candidate_vs_center = {variant: Counter() for variant in VARIANTS}
    candidate_vs_selected = {variant: Counter() for variant in VARIANTS}
    image_count = 0
    row_count = 0

    for path in paths:
        with open(path) as source:
            rows = [json.loads(line) for line in source if line.strip()]
        row_count += len(rows)
        for row in rows:
            image_id = int(row["question_id"])
            image_info = coco.loadImgs([image_id])[0]
            image_size = (int(image_info["width"]), int(image_info["height"]))
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=None))
            union_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
            category_masks = defaultdict(
                lambda: np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
            )
            present_names = set()
            for annotation in annotations:
                mask = coco.annToMask(annotation).astype(np.uint8)
                union_mask |= mask
                name = category_names[int(annotation["category_id"])]
                category_masks[name] |= mask
                present_names.add(name)

            stats = row["saver_diagnostic_stats"]
            examples = stats.get("examples", [])
            if len(examples) != stats.get("map_steps"):
                raise ValueError(
                    f"{image_id}: retained {len(examples)} examples for {stats.get('map_steps')} map steps"
                )
            image_count += 1
            for example in examples:
                selected_text = tokenizer.decode([int(example["selected_candidate_id"])])
                category_token = normalize_category_token(selected_text)
                candidate_present = category_token in present_names
                if int(example["selected_candidate_id"]) != int(example["vhr_candidate_id"]):
                    switch_tokens[
                        category_token if category_token else "<nonlexical>"
                    ] += 1

                event_hits = {}
                center_hit = None
                for variant in VARIANTS:
                    bbox, top_indices = variant_fields(example, variant)
                    point_x, point_y = point_from_patch(
                        top_indices[0], example["side"], image_size
                    )
                    crop_mapping = patch_bbox_to_raw_bbox(
                        bbox, example["side"], image_size, 336, (336, 336)
                    )
                    x0, y0, x1, y1 = crop_mapping["raw_bbox"]
                    metric = totals[variant]
                    metric["steps"] += 1
                    metric["any_point_hits"] += int(bool(union_mask[point_y, point_x]))
                    metric["any_crop_hits"] += int(bool(union_mask[y0:y1, x0:x1].any()))
                    if candidate_present:
                        target_mask = category_masks[category_token]
                        metric["candidate_steps"] += 1
                        metric["candidate_point_hits"] += int(
                            bool(target_mask[point_y, point_x])
                        )
                        event_hits[variant] = bool(target_mask[point_y, point_x])
                        metric["candidate_crop_hits"] += int(
                            bool(target_mask[y0:y1, x0:x1].any())
                        )
                        center_x, center_y = image_size[0] // 2, image_size[1] // 2
                        center_hit = bool(target_mask[center_y, center_x])
                        metric["center_candidate_hits"] += int(
                            center_hit
                        )
                if candidate_present:
                    for variant, hit in event_hits.items():
                        candidate_vs_center[variant][f"{int(hit)}{int(center_hit)}"] += 1
                        candidate_vs_selected[variant][
                            f"{int(hit)}{int(event_hits['selected'])}"
                        ] += 1

    summary = {}
    for variant, counts in totals.items():
        steps = counts["steps"]
        candidate_steps = counts["candidate_steps"]
        summary[variant] = {
            **dict(counts),
            "any_point_accuracy": counts["any_point_hits"] / max(1, steps),
            "any_crop_accuracy": counts["any_crop_hits"] / max(1, steps),
            "candidate_point_accuracy": counts["candidate_point_hits"]
            / max(1, candidate_steps),
            "candidate_crop_accuracy": counts["candidate_crop_hits"]
            / max(1, candidate_steps),
            "center_candidate_accuracy": counts["center_candidate_hits"]
            / max(1, candidate_steps),
        }

    result = {
        "purpose": "grounding diagnostic only; no CHAIR/effect metric",
        "inputs": paths,
        "files": len(paths),
        "rows": row_count,
        "images": image_count,
        "metrics": summary,
        "paired_candidate_pointing": {
            variant: {
                "vs_center": {
                    "transitions": dict(candidate_vs_center[variant]),
                    "mcnemar_exact_p": mcnemar_exact_pvalue(
                        candidate_vs_center[variant]["10"],
                        candidate_vs_center[variant]["01"],
                    ),
                },
                "vs_selected": {
                    "transitions": dict(candidate_vs_selected[variant]),
                    "mcnemar_exact_p": mcnemar_exact_pvalue(
                        candidate_vs_selected[variant]["10"],
                        candidate_vs_selected[variant]["01"],
                    ),
                },
            }
            for variant in VARIANTS
        },
        "switch_token_counts": switch_tokens.most_common(),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "x") as destination:
        json.dump(result, destination, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
