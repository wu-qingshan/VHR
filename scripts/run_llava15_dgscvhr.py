import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HallucinationTest


METHODS = {
    "vhr": "vhr-dg-unseen-a05-b01",
    "scvhr": "scvhr-dg-unseen-a05-b01",
    "dgscvhr": "dgscvhr-unseen-a05-b01",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run frozen VHR/SC-VHR/DG-SC-VHR on one fixed LLaVA-1.5 CHAIR split."
    )
    parser.add_argument("--model_path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ids_file", default="experiment/pdscvhr_dev_chair_ids.json")
    parser.add_argument("--output_dir", default="dgscvhr_dev_results")
    parser.add_argument("--methods", default="dgscvhr")
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    return parser.parse_args()


def load_ids(path, limit):
    with open(path, "r") as handle:
        image_ids = [int(image_id) for image_id in json.load(handle)]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Duplicate image IDs found in {path}")
    if not 1 <= limit <= len(image_ids):
        raise ValueError(f"limit must be within [1, {len(image_ids)}], got {limit}")
    return image_ids[:limit]


def model_name(model_path):
    return os.path.basename(os.path.realpath(model_path).rstrip(os.sep)).split("--")[-1]


def result_path(args, method):
    return os.path.join(
        args.output_dir,
        model_name(args.model_path),
        f"0-chair_0-{method}-{args.max_new_tokens}.json",
    )


def build_model_args(args, first_method):
    return SimpleNamespace(
        model_path=args.model_path,
        num_chunks=1,
        chunk_idx=0,
        device=args.device,
        method=first_method,
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
        max_new_tokens=args.max_new_tokens,
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
        seed=0,
        output_dir=args.output_dir,
        datasets="chair",
        chair_seeds="0",
        chair_samples=args.limit,
        chair_ids_file=args.ids_file,
    )


def main():
    args = parse_args()
    method_keys = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(method_keys) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if len(method_keys) != len(set(method_keys)):
        raise ValueError("Duplicate method names are not allowed")

    image_ids = load_ids(args.ids_file, args.limit)
    methods = [METHODS[key] for key in method_keys]
    planned_outputs = [result_path(args, method) for method in methods]
    existing = [path for path in planned_outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite or silently skip an existing output: " + ", ".join(existing)
        )

    print(
        "Frozen config: LLaVA-1.5, layers=1+last14, magnitude top50%, "
        "ratio=2, filter, alpha=0.5, beta=0.1"
    )
    print(f"IDs: {args.ids_file}; samples={len(image_ids)}")
    for path in planned_outputs:
        print(f"Planned output: {path}")

    test = HallucinationTest(build_model_args(args, methods[0]))
    test.output_dir = args.output_dir
    test.chair_ids = image_ids
    test.chair_samples = len(image_ids)
    for method in methods:
        test.method = method
        test.infer_dataset("chair", 0)


if __name__ == "__main__":
    main()
