import argparse
import json
import os
import sys
from math import ceil

import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation import replace_generation
from utils import get_layers, get_model


replace_generation()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-force frozen base/VHR/SC logits on fixed COCO-train captions. "
            "This is a training-feasibility diagnostic and writes no model weights."
        )
    )
    parser.add_argument("--model_path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ids_file", default="experiment/cird_teacher_feasibility_train_ids.json"
    )
    parser.add_argument(
        "--captions", default="data/coco/annotations/captions_train2014.json"
    )
    parser.add_argument("--image_dir", default="data/coco/train2014")
    parser.add_argument("--output_dir", default="cird_teacher_feasibility_results")
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--num_chunks", default=1, type=int)
    parser.add_argument("--chunk_idx", default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_chunks <= 0 or not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError("Require num_chunks>0 and chunk_idx in [0,num_chunks)")
    with open(args.ids_file) as source:
        image_ids = [int(image_id) for image_id in json.load(source)]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate teacher-feasibility image IDs")
    if not 1 <= args.limit <= len(image_ids):
        raise ValueError(f"limit must be within [1,{len(image_ids)}]")
    image_ids = image_ids[: args.limit]

    model_name = os.path.basename(os.path.realpath(args.model_path).rstrip(os.sep)).split("--")[-1]
    output_path = os.path.join(
        args.output_dir,
        model_name,
        f"{args.chunk_idx}-cird-teacher-feasibility.jsonl",
    )
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    coco = COCO(args.captions)
    device = torch.device(args.device)
    processor, model, template, _, num_img_tokens, image_token_id = get_model(
        args.model_path, device, "sdpa"
    )
    if "llava-1.5" not in model_name:
        raise ValueError("CIRD teacher feasibility is currently locked to LLaVA-1.5")
    target_layers = [1] + list(range(len(get_layers(model))))[-14:]

    chunk_size = ceil(len(image_ids) / args.num_chunks)
    chunk_ids = image_ids[
        args.chunk_idx * chunk_size : (args.chunk_idx + 1) * chunk_size
    ]
    print(json.dumps({
        "purpose": "CIRD teacher-forcing feasibility only; no training",
        "ids_file": args.ids_file,
        "chunk_idx": args.chunk_idx,
        "num_chunks": args.num_chunks,
        "images": len(chunk_ids),
        "output": output_path,
    }, indent=2))

    with open(output_path, "x") as destination:
        for image_id in tqdm(chunk_ids, desc=f"CIRD teacher chunk {args.chunk_idx}"):
            image_info = coco.loadImgs([image_id])[0]
            annotations = sorted(
                coco.loadAnns(coco.getAnnIds(imgIds=[image_id])),
                key=lambda annotation: int(annotation["id"]),
            )
            if not annotations:
                raise RuntimeError(f"No caption annotation for image {image_id}")
            annotation = annotations[0]
            caption = annotation["caption"].strip()
            forced_ids = processor.tokenizer(
                caption, add_special_tokens=False
            ).input_ids
            forced_ids = [int(token_id) for token_id in forced_ids]
            eos_token_id = int(model.generation_config.eos_token_id)
            if not forced_ids or forced_ids[-1] != eos_token_id:
                forced_ids.append(eos_token_id)

            image = Image.open(os.path.join(args.image_dir, image_info["file_name"])).convert("RGB")
            prompt = template.format(question="Please describe this image in detail.")
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            image_positions = (inputs["input_ids"] == image_token_id).nonzero()
            if image_positions.shape[0] != 1:
                raise RuntimeError(f"Expected one image placeholder, got {image_positions.shape[0]}")
            image_start = int(image_positions[0, 1].item())

            generate_kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": len(forced_ids),
                "vhr_layers": target_layers,
                "vhr_aug_ratio": 2.0,
                "vhr_filter": True,
                "vhr_score_mode": "magnitude",
                "vhr_head_fraction": 0.5,
                "vhr_positive_only": False,
                "vhr_contrast_mode": "remove",
                "vhr_image_start": image_start,
                "vhr_image_end": image_start + 1,
                "vhr_base_contrast": True,
                "vhr_logit_alpha": 0.5,
                "vhr_logit_beta": 0.1,
                "vhr_teacher_diagnostic": True,
                "vhr_forced_token_ids": forced_ids,
                "return_dict_in_generate": True,
            }
            with torch.inference_mode():
                output = model.generate(**inputs, **generate_kwargs)
            generated_suffix = [
                int(token_id) for token_id in output.sequences[0, -len(forced_ids) :].tolist()
            ]
            if generated_suffix != forced_ids:
                raise RuntimeError(f"Forced token mismatch for image {image_id}")
            stats = dict(model._last_cird_teacher_stats)
            if stats["token_count"] != len(forced_ids):
                raise RuntimeError(f"Teacher stat length mismatch for image {image_id}")
            decoded = processor.tokenizer.decode(forced_ids, skip_special_tokens=True)
            if decoded.strip() != caption:
                raise RuntimeError(
                    f"Caption tokenization roundtrip mismatch for image {image_id}: {decoded!r}"
                )
            destination.write(json.dumps({
                "image_id": image_id,
                "caption_annotation_id": int(annotation["id"]),
                "caption": caption,
                "forced_token_ids": forced_ids,
                "teacher_stats": stats,
            }) + "\n")


if __name__ == "__main__":
    main()
