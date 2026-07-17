import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HallucinationTest


VHR_METHOD = "vhr-h18-rr-sameprocess-smoke"
SC_METHOD = "scvhr-h18-rr-sameprocess-smoke"
RR_METHOD = "rrscvhr-h18-a05-b01-sameprocess-smoke"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one CHAIR and a small POPE RR-SC-VHR equivalence smoke test."
    )
    parser.add_argument("--model_path", default="models/instructblip-vicuna-7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="rrscvhr_smoke_results")
    parser.add_argument("--pope_limit", default=10, type=int)
    return parser.parse_args()


def build_args(args):
    return SimpleNamespace(
        model_path=args.model_path,
        num_chunks=1,
        chunk_idx=0,
        device=args.device,
        method=SC_METHOD,
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
        vhr_logit_alpha=0.5,
        vhr_logit_beta=0.1,
        max_new_tokens=512,
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
        seed=0,
        output_dir=args.output_dir,
        datasets="chair,pope_popular,pope_adversarial,pope_random",
        chair_seeds="0",
        chair_samples=500,
        chair_ids_file=None,
        dataset_limit=1,
    )


def main():
    args = parse_args()
    test = HallucinationTest(build_args(args))
    test.dataset_limit = 1
    test.method = SC_METHOD
    test.infer_dataset("chair", 0)
    test.method = RR_METHOD
    test.infer_dataset("chair", 0)
    test.dataset_limit = args.pope_limit
    test.method = VHR_METHOD
    for dataset in ("pope_popular", "pope_adversarial", "pope_random"):
        test.infer_dataset(dataset, 0)
    test.method = RR_METHOD
    for dataset in ("pope_popular", "pope_adversarial", "pope_random"):
        test.infer_dataset(dataset, 0)


if __name__ == "__main__":
    main()
