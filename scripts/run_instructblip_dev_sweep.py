import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HallucinationTest


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed InstructBLIP CS-VHR development sweep with one model load.")
    parser.add_argument("--model_path", default="models/instructblip-vicuna-7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="dev_results")
    parser.add_argument("--chair_ids_file", default="experiment/dev_chair_ids.json")
    parser.add_argument("--max_new_tokens", default=512, type=int)
    return parser.parse_args()


def build_model_args(args):
    return SimpleNamespace(
        model_path=args.model_path,
        num_chunks=1,
        chunk_idx=0,
        device=args.device,
        method="vhr-dev-h18",
        vhr_aug_ratio=2.0,
        vhr_layers=18,
        vhr_layer1=False,
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
    test = HallucinationTest(build_model_args(args))
    configs = [
        ("vhr-dev-h18", "magnitude", 0.5, False),
        ("csvhr-dir50-h18", "directional", 0.5, True),
        ("csvhr-dir25-h18", "directional", 0.25, True),
    ]
    for method, score_mode, head_fraction, positive_only in configs:
        test.method = method
        test.vhr_score_mode = score_mode
        test.vhr_head_fraction = head_fraction
        test.vhr_positive_only = positive_only
        test.infer_dataset("chair", 0)


if __name__ == "__main__":
    main()
