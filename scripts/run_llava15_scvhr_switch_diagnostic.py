import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HallucinationTest


METHOD = "scvhrdiag-unseen-a05-b01"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerun frozen LLaVA SC-VHR while recording token-switch diagnostics."
    )
    parser.add_argument("--model_path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ids_file", default="experiment/pdscvhr_dev_chair_ids.json")
    parser.add_argument("--output_dir", default="scvhr_switch_diagnostic_results")
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    return parser.parse_args()


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
    image_ids = [int(image_id) for image_id in json.load(open(args.ids_file))]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image IDs")
    if not 1 <= args.limit <= len(image_ids):
        raise ValueError(f"limit must be within [1, {len(image_ids)}]")
    image_ids = image_ids[: args.limit]

    model_name = os.path.basename(os.path.realpath(args.model_path).rstrip(os.sep)).split("--")[-1]
    output_path = os.path.join(
        args.output_dir,
        model_name,
        f"0-chair_0-{METHOD}-{args.max_new_tokens}.json",
    )
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite or skip {output_path}")

    print(f"Planned output: {output_path}")
    test = HallucinationTest(build_model_args(args))
    test.chair_ids = image_ids
    test.chair_samples = len(image_ids)
    test.infer_dataset("chair", 0)


if __name__ == "__main__":
    main()
