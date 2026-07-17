import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HallucinationTest


METHODS = {
    "vhr": "vhr-vcrp-mechanism-h14",
    "vcrpdiagvhr": "vcrpdiagvhr-mechanism-h14",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen VHR and no-intervention VCRP diagnostics on the fixed "
            "PD development split. This runner cannot launch VCRP effects."
        )
    )
    parser.add_argument("--model_path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ids_file", default="experiment/pdscvhr_dev_chair_ids.json")
    parser.add_argument("--output_dir", default="vcrp_diagnostic_results")
    parser.add_argument("--methods", default="vhr,vcrpdiagvhr")
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--num_chunks", default=1, type=int)
    parser.add_argument("--chunk_idx", default=0, type=int)
    return parser.parse_args()


def load_ids(path, limit):
    with open(path) as source:
        image_ids = [int(image_id) for image_id in json.load(source)]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Duplicate image IDs in {path}")
    if not 1 <= limit <= len(image_ids):
        raise ValueError(f"limit must be within [1, {len(image_ids)}]")
    return image_ids[:limit]


def build_model_args(args, method):
    return SimpleNamespace(
        model_path=args.model_path,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
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
        datasets="chair",
        chair_seeds="0",
        chair_samples=args.limit,
        chair_ids_file=args.ids_file,
        dataset_limit=None,
    )


def output_path(args, method):
    model_name = os.path.basename(os.path.realpath(args.model_path).rstrip(os.sep)).split("--")[-1]
    return os.path.join(
        args.output_dir,
        model_name,
        f"{args.chunk_idx}-chair_0-{method}-{args.max_new_tokens}.json",
    )


def main():
    args = parse_args()
    if args.num_chunks <= 0 or not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError("Require num_chunks>0 and chunk_idx in [0,num_chunks)")
    method_keys = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(method_keys) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown or effect-capable methods are forbidden: {sorted(unknown)}")
    if len(method_keys) != len(set(method_keys)):
        raise ValueError("Duplicate methods are not allowed")

    image_ids = load_ids(args.ids_file, args.limit)
    methods = [METHODS[key] for key in method_keys]
    outputs = [output_path(args, method) for method in methods]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    print(json.dumps({
        "purpose": "VCRP mechanism diagnostic only; no intervention",
        "ids_file": args.ids_file,
        "sample_count": len(image_ids),
        "num_chunks": args.num_chunks,
        "chunk_idx": args.chunk_idx,
        "methods": method_keys,
        "outputs": outputs,
    }, indent=2))
    test = HallucinationTest(build_model_args(args, methods[0]))
    test.chair_ids = image_ids
    test.chair_samples = len(image_ids)
    for method in methods:
        test.method = method
        test.infer_dataset("chair", 0)


if __name__ == "__main__":
    main()
