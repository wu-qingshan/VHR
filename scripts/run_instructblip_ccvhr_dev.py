import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HallucinationTest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed InstructBLIP CC-VHR development comparison with one model load."
    )
    parser.add_argument("--model_path", default="models/instructblip-vicuna-7b")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output_dir", default="ccvhr_dev_results")
    parser.add_argument("--chair_ids_file", default="experiment/dev_chair_ids.json")
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--noise_step", default=500, type=int)
    parser.add_argument(
        "--variants",
        default="remove,zero",
        help="comma-separated text counterfactuals to run (remove and/or zero)",
    )
    return parser.parse_args()


def build_model_args(args):
    return SimpleNamespace(
        model_path=args.model_path,
        num_chunks=1,
        chunk_idx=0,
        device=args.device,
        method="ccvhr-consensus-remove-h18",
        vhr_aug_ratio=2.0,
        vhr_layers=18,
        vhr_layer1=False,
        vhr_filter=True,
        vhr_tau=1.0,
        vhr_alpha=0.5,
        vhr_rho_max=2.0,
        vhr_score_mode="consensus",
        vhr_head_fraction=0.5,
        vhr_positive_only=False,
        vhr_binary_fallback=False,
        vhr_contrast_mode="remove",
        vhr_directional_delay=64,
        vhr_noise_step=args.noise_step,
        cd_alpha=0.5,
        cd_beta=0.1,
        max_new_tokens=args.max_new_tokens,
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
        seed=0,
        output_dir=args.output_dir,
        datasets="chair",
        chair_seeds="0",
        chair_samples=100,
        chair_ids_file=args.chair_ids_file,
    )


def main():
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = set(variants) - {"remove", "zero"}
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    test = HallucinationTest(build_model_args(args))
    for contrast_mode in variants:
        test.method = f"ccvhr-consensus-{contrast_mode}-h18-n{args.noise_step}"
        test.vhr_score_mode = "consensus"
        test.vhr_head_fraction = 0.5
        test.vhr_positive_only = False
        test.vhr_contrast_mode = contrast_mode
        test.infer_dataset("chair", 0)


if __name__ == "__main__":
    main()
