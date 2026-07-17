import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench
from main import HallucinationTest
from saver import patch_bbox_to_raw_bbox
from scripts.run_llava15_carescvhr import METHODS, build_model_args
from utils import get_layers


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one isolated SAVER engineering diagnostic: map a selected-head "
            "patch box to the raw image, process the crop, and execute a base "
            "forward with the exact generated-token prefix. No effect metric is computed."
        )
    )
    parser.add_argument("--model_path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_id", default=33532, type=int)
    parser.add_argument("--example_index", default=0, type=int)
    parser.add_argument(
        "--map_source", choices=("switch", "all"), default="switch"
    )
    parser.add_argument(
        "--map_variant",
        choices=("selected", "all_head", "content", "value"),
        default="selected",
    )
    parser.add_argument("--max_new_tokens", default=96, type=int)
    parser.add_argument("--output_dir", default="saver_local_diagnostic_results")
    return parser.parse_args()


def append_generated_prefix(processed, generated_ids, device):
    inputs = {key: value.to(device) for key, value in processed.items()}
    generated_ids = generated_ids.to(device)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], generated_ids], dim=1)
    if "attention_mask" in inputs:
        suffix_mask = torch.ones_like(generated_ids, device=device)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], suffix_mask], dim=1
        )
    return inputs


def transient_state_summary(model):
    attributes = (
        "attn_output_tc",
        "attn_output_nc",
        "aug_heads",
        "aug_ratio_per_head",
        "vhr_baseline_heads",
        "vhr_directional_heads",
        "vhr_phase",
        "vhr_saver_last_map",
        "vhr_saver_last_all_head_map",
        "vhr_saver_last_content_map",
        "vhr_saver_last_value_map",
        "vhr_saver_image_keys_content",
    )
    nonempty = []
    for layer_index, layer in enumerate(get_layers(model)):
        attention = layer.self_attn
        for attribute in attributes:
            value = getattr(attention, attribute, None)
            if value is not None and value is not False:
                nonempty.append([layer_index, attribute])
    return {"clean": not nonempty, "nonempty": nonempty}


def main():
    args = parse_args()
    if args.example_index < 0:
        raise ValueError("example_index must be non-negative")
    if args.max_new_tokens <= args.example_index:
        raise ValueError("max_new_tokens must exceed example_index")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "diagnostic.json")
    crop_path = os.path.join(args.output_dir, f"coco_{args.image_id}_crop.png")
    if os.path.exists(manifest_path) or os.path.exists(crop_path):
        raise FileExistsError(
            f"Refusing to overwrite existing SAVER diagnostic outputs in {args.output_dir}"
        )

    runner_args = SimpleNamespace(
        model_path=args.model_path,
        device=args.device,
        ids_file=None,
        output_dir=args.output_dir,
        dataset="chair",
        limit=1,
        max_new_tokens=args.max_new_tokens,
    )
    method = METHODS["saverdiagscvhr"]
    test = HallucinationTest(build_model_args(runner_args, method))
    if "llava-1.5" not in test.model_name:
        raise ValueError("This coordinate diagnostic is currently locked to LLaVA-1.5")

    dataset = CHAIRBench(image_ids=[args.image_id])
    image, question, _ = dataset[args.image_id]
    image = image.convert("RGB")
    prompt = test.template.format(question=question)
    processed = test.processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(test.device) for key, value in processed.items()}
    prompt_token_count = inputs["input_ids"].shape[1]
    image_start, _, _, image_token_end = test.get_image_position(inputs["input_ids"])
    target_layers = [1] + list(range(len(get_layers(test.model))))[-14:]

    generate_kwargs = {
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "vhr_aug_ratio": 2.0,
        "vhr_layers": target_layers,
        "vhr_filter": True,
        "vhr_score_mode": "magnitude",
        "vhr_head_fraction": 0.5,
        "vhr_positive_only": False,
        "vhr_contrast_mode": "remove",
        "vhr_base_contrast": True,
        "vhr_saver_diagnostic": True,
        "vhr_logit_alpha": 0.5,
        "vhr_logit_beta": 0.1,
        "vhr_image_start": image_start,
        "vhr_image_end": image_token_end,
        "vhr_image_count": test.num_img_tokens,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    with torch.inference_mode():
        generated = test.model.generate(**inputs, **generate_kwargs)

    if not torch.equal(
        generated.sequences[:, :prompt_token_count], inputs["input_ids"]
    ):
        raise RuntimeError("Generated sequence does not preserve the original prompt token IDs")
    diagnostic = dict(test.model._last_vhr_saver_diagnostic_stats)
    example_key = "switch_examples" if args.map_source == "switch" else "examples"
    examples = diagnostic.get(example_key, [])
    if args.example_index >= len(examples):
        raise RuntimeError(
            f"Requested {args.map_source} example {args.example_index}, "
            f"but only {len(examples)} maps were retained"
        )
    map_example = examples[args.example_index]
    step_index = int(map_example["step_index"])
    if step_index >= args.max_new_tokens:
        raise RuntimeError(
            f"Map step {step_index} is outside the generated sequence of length {args.max_new_tokens}"
        )

    image_processor = test.processor.image_processor
    shortest_edge = int(image_processor.size["shortest_edge"])
    crop_size = (
        int(image_processor.crop_size["width"]),
        int(image_processor.crop_size["height"]),
    )
    bbox_key = "bbox" if args.map_variant == "selected" else f"{args.map_variant}_bbox"
    mapping = patch_bbox_to_raw_bbox(
        map_example[bbox_key],
        map_example["side"],
        image.size,
        shortest_edge,
        crop_size,
    )
    local_image = image.crop(tuple(mapping["raw_bbox"]))
    local_image.save(crop_path)

    generated_prefix = generated.sequences[
        :, prompt_token_count : prompt_token_count + step_index
    ]
    full_processed = test.processor(text=prompt, images=image, return_tensors="pt")
    local_processed = test.processor(text=prompt, images=local_image, return_tensors="pt")
    full_inputs = append_generated_prefix(full_processed, generated_prefix, test.device)
    local_inputs = append_generated_prefix(local_processed, generated_prefix, test.device)

    cleanup_after_generate = transient_state_summary(test.model)
    with torch.inference_mode():
        full_output = test.model(**full_inputs, use_cache=False, return_dict=True)
        local_output = test.model(**local_inputs, use_cache=False, return_dict=True)
    full_logits = full_output.logits[0, -1].float()
    local_logits = local_output.logits[0, -1].float()
    if not torch.isfinite(full_logits).all() or not torch.isfinite(local_logits).all():
        raise RuntimeError("Full-image or crop local-forward logits contain non-finite values")
    cleanup_after_local_forward = transient_state_summary(test.model)

    full_argmax = int(full_logits.argmax().item())
    local_argmax = int(local_logits.argmax().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(full_logits, local_logits, dim=0).item()
    )
    result = {
        "purpose": "engineering-only; no CHAIR or effect metric",
        "model": test.model_name,
        "image_id": args.image_id,
        "raw_image_size": list(image.size),
        "map_source": args.map_source,
        "map_variant": args.map_variant,
        "map_example_index": args.example_index,
        "map_example": map_example,
        "mapping": mapping,
        "saved_crop": crop_path,
        "saved_crop_size": list(local_image.size),
        "processor_pixel_shape": list(local_inputs["pixel_values"].shape),
        "prompt_token_count": prompt_token_count,
        "generated_prefix_token_count": int(generated_prefix.shape[1]),
        "full_prefix_token_count": int(local_inputs["input_ids"].shape[1]),
        "full_logits_shape": list(full_logits.shape),
        "local_logits_shape": list(local_logits.shape),
        "full_argmax_id": full_argmax,
        "local_argmax_id": local_argmax,
        "full_argmax_text": test.processor.tokenizer.decode([full_argmax]),
        "local_argmax_text": test.processor.tokenizer.decode([local_argmax]),
        "full_local_logit_cosine": cosine,
        "cleanup_after_generate": cleanup_after_generate,
        "cleanup_after_local_forward": cleanup_after_local_forward,
    }
    if args.map_source == "switch":
        vhr_candidate = int(map_example["vhr_candidate_id"])
        base_candidate = int(map_example["base_candidate_id"])
        selected_candidate = int(map_example["selected_candidate_id"])
        result["switch_candidates"] = {
            "vhr": {
                "id": vhr_candidate,
                "text": test.processor.tokenizer.decode([vhr_candidate]),
            },
            "base": {
                "id": base_candidate,
                "text": test.processor.tokenizer.decode([base_candidate]),
            },
            "selected": {
                "id": selected_candidate,
                "text": test.processor.tokenizer.decode([selected_candidate]),
            },
        }
        result["candidate_pair_support"] = {
            "full_selected_minus_vhr": float(
                (full_logits[selected_candidate] - full_logits[vhr_candidate]).item()
            ),
            "local_selected_minus_vhr": float(
                (local_logits[selected_candidate] - local_logits[vhr_candidate]).item()
            ),
            "full_selected_minus_base": float(
                (full_logits[selected_candidate] - full_logits[base_candidate]).item()
            ),
            "local_selected_minus_base": float(
                (local_logits[selected_candidate] - local_logits[base_candidate]).item()
            ),
        }
    with open(manifest_path, "x") as destination:
        json.dump(result, destination, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
