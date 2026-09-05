#!/usr/bin/env python3
"""Build the E0.2 twenty-image human annotation material (frozen sampling)."""

import argparse
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def atomic_write(path, text):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sample_images(tasks, seed=2026090401, count=20):
    image_ids = sorted({task["image_id"] for task in tasks})
    return sorted(random.Random(seed).sample(image_ids, count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="qwen_vera_dev_results/tasks.json")
    parser.add_argument(
        "--extractions", default="qwen_vera_dev_results/source/extractions.json"
    )
    parser.add_argument(
        "--verifier", default="qwen_vera_dev_results/source/verifier.json"
    )
    parser.add_argument("--output_dir", default="mgr_ceiling_results")
    parser.add_argument("--seed", type=int, default=2026090401)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    tasks = json.load(open(args.tasks))
    extractions = {row["image_id"]: row for row in json.load(open(args.extractions))}
    scores = {row["query_id"]: float(row["score"]) for row in load_jsonl(args.verifier)}
    sample = sample_images(tasks, args.seed, args.count)
    selected = [task for task in tasks if task["image_id"] in set(sample)]

    phrase_scores = {}
    for image_id in sample:
        for candidate in extractions[image_id]["candidates"]:
            score = scores[candidate["query_id"]]
            phrase_scores[(image_id, candidate["phrase"])] = (
                candidate["root_lemma"],
                score,
            )
            for occurrence in candidate["occurrences"]:
                phrase_scores.setdefault(
                    (image_id, occurrence["phrase"]), (candidate["root_lemma"], score)
                )

    output_dir = Path(args.output_dir)
    sampling_record = {
        "population": "image ids with at least one VERA verifier trigger sentence",
        "population_size": len({task["image_id"] for task in tasks}),
        "rule": f"random.Random({args.seed}).sample(sorted(image_ids), {args.count})",
        "sample": sample,
        "triggered_sentences": len(selected),
        "trigger_phrases": sum(len(task["trigger_phrases"]) for task in selected),
    }
    atomic_write(
        output_dir / "e02_sampling.json",
        json.dumps(sampling_record, indent=2, ensure_ascii=False) + "\n",
    )

    form_rows = []
    lines = [
        "# E0.2 二十图人工触发审计材料",
        "",
        "抽样规则（冻结）：`random.Random(2026090401).sample(sorted(触发图), 20)`。",
        f"共 {len(selected)} 个触发句、{sampling_record['trigger_phrases']} 个触发 phrase。",
        "",
        "对每个 trigger phrase 独立三分类（只看图与 phrase，不看 CHAIR 标签）：",
        "",
        "1. **真幻觉**：图中确无该实体；",
        "2. **真实物体误触发**：图中存在该实体，verifier 误报；",
        "3. **非实体词误触发**：抽象词/属性/短语边界错误等（含 parser artifact）。",
        "",
        "图片文件：`val2014/COCO_val2014_{image_id:012d}.jpg`（仓库根目录相对路径）。",
        "标注结果填入 `e02_annotation_form.csv` 的 label 列（1/2/3）。",
        "",
    ]
    for image_id in sample:
        image_tasks = [t for t in selected if t["image_id"] == image_id]
        lines.append(f"## 图 {image_id}")
        lines.append("")
        lines.append(
            f"![{image_id}](/home/nlp/VHR/val2014/COCO_val2014_{image_id:012d}.jpg)"
        )
        lines.append("")
        if not image_tasks:
            lines.append("（该图无触发句——不应发生，仅兜底）")
            lines.append("")
            continue
        for task in sorted(image_tasks, key=lambda t: t["sentence_index"]):
            lines.append(
                f"**句 {task['sentence_index']}**（verifier 分数 ≤ 0.1 触发）："
            )
            lines.append(f"> {task['source_sentence'].strip()}")
            lines.append("")
            lines.append("| trigger phrase | root | verifier 分数 | 标注（1/2/3） |")
            lines.append("|---|---|---:|---|")
            for phrase in task["trigger_phrases"]:
                root, score = phrase_scores[(image_id, phrase)]
                lines.append(f"| {phrase} | {root} | {score:.3f} | |")
                form_rows.append(
                    {
                        "image_id": image_id,
                        "sentence_index": task["sentence_index"],
                        "trigger_phrase": phrase,
                        "root_lemma": root,
                        "verifier_score": round(score, 6),
                        "label": "",
                        "notes": "",
                    }
                )
            lines.append("")
    atomic_write(output_dir / "e02_annotation_material.md", "\n".join(lines))

    csv_lines = [
        "image_id,sentence_index,trigger_phrase,root_lemma,verifier_score,label,notes"
    ]
    for row in form_rows:
        phrase = row["trigger_phrase"].replace('"', '""')
        notes = ""
        csv_lines.append(
            f"{row['image_id']},{row['sentence_index']},\"{phrase}\","
            f"{row['root_lemma']},{row['verifier_score']},{row['label']},{notes}"
        )
    atomic_write(output_dir / "e02_annotation_form.csv", "\n".join(csv_lines) + "\n")

    print(
        json.dumps(
            {
                "sample": sample,
                "triggered_sentences": len(selected),
                "trigger_phrases": len(form_rows),
                "outputs": [
                    str(output_dir / "e02_sampling.json"),
                    str(output_dir / "e02_annotation_material.md"),
                    str(output_dir / "e02_annotation_form.csv"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
