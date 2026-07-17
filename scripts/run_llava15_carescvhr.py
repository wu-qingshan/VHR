import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HallucinationTest


METHODS = {
    "vhr": "vhr-care-dev-h14",
    "scvhr": "scvhr-care-dev-h14-a05-b01",
    "cascvhr": "cascvhr-care-dev-h14-a05-b01",
    "carescvhr": "carescvhr-care-dev-h14-a05-b01",
    "saverdiagscvhr": "saverdiagscvhr-engineering-h14-a05-b01",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run frozen VHR/SC/CA-SC/CARE-SC on an isolated LLaVA-1.5 CHAIR split."
    )
    parser.add_argument("--model_path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ids_file", default="experiment/care_dev_chair_ids.json")
    parser.add_argument("--output_dir", default="care_dev_results")
    parser.add_argument("--methods", default="vhr,scvhr,cascvhr,carescvhr")
    parser.add_argument("--dataset", choices=("chair", "pope_popular"), default="chair")
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    return parser.parse_args()


def build_model_args(args, method):
    return SimpleNamespace(
        model_path=args.model_path,
        num_chunks=1,
        chunk_idx=0,
        device=args.device,
        method=method,
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
        datasets=args.dataset,
        chair_seeds="0",
        chair_samples=args.limit,
        chair_ids_file=args.ids_file,
        dataset_limit=args.limit if args.dataset != "chair" else None,
    )


def main():
    args = parse_args()
    method_keys = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(method_keys) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if len(method_keys) != len(set(method_keys)):
        raise ValueError("Duplicate methods are not allowed")

    image_ids = None
    if args.dataset == "chair":
        with open(args.ids_file) as source:
            image_ids = [int(image_id) for image_id in json.load(source)]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("Duplicate image IDs")
        if not 1 <= args.limit <= len(image_ids):
            raise ValueError(f"limit must be within [1, {len(image_ids)}]")
        image_ids = image_ids[: args.limit]
    elif args.limit <= 0:
        raise ValueError("limit must be positive")

    model_name = os.path.basename(os.path.realpath(args.model_path).rstrip(os.sep)).split("--")[-1]
    methods = [METHODS[key] for key in method_keys]
    outputs = [
        os.path.join(
            args.output_dir,
            model_name,
            f"0-{args.dataset}_0-{method}-{args.max_new_tokens}.json",
        )
        for method in methods
    ]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError("Refusing to overwrite or skip: " + ", ".join(existing))

    print(json.dumps({
        "dataset": args.dataset,
        "ids_file": args.ids_file if args.dataset == "chair" else None,
        "sample_count": args.limit,
        "methods": method_keys,
        "outputs": outputs,
    }, indent=2))
    test = HallucinationTest(build_model_args(args, methods[0]))
    if args.dataset == "chair":
        test.chair_ids = image_ids
        test.chair_samples = len(image_ids)
    for method in methods:
        test.method = method
        test.infer_dataset(args.dataset, 0)


if __name__ == "__main__":
    main()
