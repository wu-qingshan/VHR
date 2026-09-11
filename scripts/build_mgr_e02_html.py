#!/usr/bin/env python3
"""Build a self-contained HTML viewing aid for the E0.2 human annotation."""

import base64
import csv
import json
from pathlib import Path


FORM = Path("mgr_ceiling_results/e02_annotation_form.csv")
TASKS = Path("qwen_vera_dev_results/tasks.json")
OUTPUT = Path("mgr_ceiling_results/e02_annotation_view.html")


def image_data_uri(image_id):
    path = Path(f"val2014/COCO_val2014_{int(image_id):012d}.jpg")
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{data}"


def main():
    rows = list(csv.DictReader(FORM.open(newline="")))
    if len(rows) != 38:
        raise ValueError(f"expected 38 rows, got {len(rows)}")
    sentence_by_key = {}
    for task in json.load(TASKS.open()):
        for phrase in task["trigger_phrases"]:
            sentence_by_key[(int(task["image_id"]), phrase)] = task["source_sentence"].strip()

    entries = []
    for row in rows:
        image_id = int(row["image_id"])
        phrase = row["trigger_phrase"]
        sentence = sentence_by_key.get((image_id, phrase))
        if sentence is None:
            raise ValueError(f"sentence not found for {image_id} / {phrase!r}")
        entries.append({"image_id": image_id, "sentence": sentence, "phrase": phrase})

    images = sorted({entry["image_id"] for entry in entries})
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>E0.2 annotation viewer</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#fafafa;color:#222}",
        "img{max-width:680px;max-height:560px;border:1px solid #999;display:block;margin:8px 0}",
        ".sent{background:#fff;border-left:4px solid #888;padding:6px 10px;margin:8px 0}",
        ".phrase{font-weight:bold}",
        ".hint{color:#666}",
        "li{margin:4px 0}</style></head><body>",
        "<h1>E0.2 触发人工审计（38 phrase）</h1>",
        "<p class='hint'>只看图与 phrase 判断，忽略 verifier 分数。正式标注填 e02_annotation_form.csv 的 label 列："
        "1=真幻觉，2=真实物体误触发，3=非实体词误触发。拿不准的在 notes 写一句。</p>",
    ]
    for image_id in images:
        html.append(f"<h2>图 {image_id}</h2>")
        html.append(f"<img src='{image_data_uri(image_id)}'>")
        current = None
        for entry in (e for e in entries if e["image_id"] == image_id):
            if entry["sentence"] != current:
                current = entry["sentence"]
                html.append(f"<div class='sent'>{current}</div>")
            html.append(f"<li><span class='phrase'>{entry['phrase']}</span></li>")
    html.append("</ul></body></html>")
    OUTPUT.write_text("\n".join(html))
    print(f"wrote {OUTPUT} with {len(entries)} phrases over {len(images)} images")


if __name__ == "__main__":
    main()
