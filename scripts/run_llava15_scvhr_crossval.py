import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HallucinationTest


METHOD = "scvhr-llava15-frozen-a05-b01"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered LLaVA-1.5 SC-VHR dev then validation evaluation "
            "with one model load and no parameter sweep."
        )
    )
    parser.add_argument("--model_path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dev_ids", default="experiment/dev_chair_ids.json")
    parser.add_argument("--validation_ids", default="experiment/validation_chair_ids.json")
    parser.add_argument("--dev_output_dir", default="llava15_scvhr_dev_results")
    parser.add_argument("--validation_output_dir", default="llava15_scvhr_validation_results")
    return parser.parse_args()


def load_ids(path):
    with open(path, "r") as handle:
        image_ids = json.load(handle)
    if not isinstance(image_ids, list) or len(image_ids) != 100:
        raise ValueError(f"Expected exactly 100 image IDs in {path}, got {len(image_ids)}")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"Duplicate image IDs found in {path}")
    return image_ids


def result_path(output_dir, model_path):
    model_name = os.path.basename(os.path.realpath(model_path).rstrip(os.sep)).split("--")[-1]
    return os.path.join(output_dir, model_name, f"0-chair_0-{METHOD}-512.json")


def build_model_args(args):
    return SimpleNamespace(
        model_path=args.model_path,
        num_chunks=1,
        chunk_idx=0,
        device=args.device,
        method=METHOD,
        vhr_aug_ratio=2.0,
        vhr_layers=14,
        vhr_layer1=True,
        vhr_filter=True,
        vhr_tau=1.0,
        vhr_alpha=0.5,
        vhr_rho_max=2.0,
        vhr_score_mode="magnitude",
        vhr_head_fraction=0.5,
        vhr_positive_only=False,
        vhr_binary_fallback=False,
        vhr_contrast_mode="remove",
        vhr_directional_delay=64,
        vhr_noise_step=500,
        cd_alpha=0.5,
        cd_beta=0.1,
        vhr_logit_alpha=0.5,
        vhr_logit_beta=0.1,
        max_new_tokens=512,
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
        seed=0,
        output_dir=args.dev_output_dir,
        datasets="chair",
        chair_seeds="0",
        chair_samples=100,
        chair_ids_file=args.dev_ids,
    )


def main():
    args = parse_args()
    dev_ids = load_ids(args.dev_ids)
    validation_ids = load_ids(args.validation_ids)
    overlap = set(dev_ids) & set(validation_ids)
    if overlap:
        raise ValueError(f"Dev/validation ID overlap: {sorted(overlap)[:10]}")

    planned_outputs = [
        result_path(args.dev_output_dir, args.model_path),
        result_path(args.validation_output_dir, args.model_path),
    ]
    existing = [path for path in planned_outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite or silently skip an existing output: " + ", ".join(existing)
        )

    print("Frozen SC-VHR config: layers=1+last14, magnitude top50%, ratio=2, filter, alpha=0.5, beta=0.1")
    print(f"Dev output: {planned_outputs[0]}")
    print(f"Validation output: {planned_outputs[1]}")

    test = HallucinationTest(build_model_args(args))
    test.method = METHOD

    test.output_dir = args.dev_output_dir
    test.chair_ids = dev_ids
    test.chair_samples = len(dev_ids)
    test.infer_dataset("chair", 0)

    test.output_dir = args.validation_output_dir
    test.chair_ids = validation_ids
    test.chair_samples = len(validation_ids)
    test.infer_dataset("chair", 0)


if __name__ == "__main__":
    main()
