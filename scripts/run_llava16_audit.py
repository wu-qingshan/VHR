#!/usr/bin/env python3
"""Run the preregistered LLaVA-1.6 transfer audit (vanilla / VHR / SC-VHR)."""

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HallucinationTest


MODEL_PATH = "models/llava-v1.6-vicuna-7b-hf"
METHOD_TAGS = {
    "vanilla": "llava16-vanilla-512",
    "vhr": "vhr-l1h14",
    "scvhr": "scvhr-llava16-audit-a05-b01",
}
FROZEN_NOTE = (
    "Frozen audit config: layers=1+last14, magnitude top50%, ratio=2, filter, "
    "sc alpha=0.5 beta=0.1, greedy 512"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one preregistered method on the fixed dev and validation CHAIR "
            "sets with one model load (AUDIT-PLAN section 3)."
        )
    )
    parser.add_argument("--method", required=True, choices=sorted(METHOD_TAGS))
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dev_ids", default="experiment/dev_chair_ids.json")
    parser.add_argument(
        "--validation_ids", default="experiment/validation_chair_ids.json"
    )
    parser.add_argument("--dev_output_dir", default="llava16_audit_dev_results")
    parser.add_argument(
        "--validation_output_dir", default="llava16_audit_validation_results"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="determinism spot check: 2 fixed dev images x 2 repeats, no files written",
    )
    return parser.parse_args()


def load_ids(path):
    with open(path, "r") as handle:
        image_ids = json.load(handle)
    if not isinstance(image_ids, list) or len(image_ids) != 100:
        raise ValueError(f"Expected exactly 100 image IDs in {path}, got {len(image_ids)}")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"Duplicate image IDs found in {path}")
    return image_ids


def result_path(output_dir, model_path, method_tag):
    model_name = (
        os.path.basename(os.path.realpath(model_path).rstrip(os.sep)).split("--")[-1]
    )
    return os.path.join(output_dir, model_name, f"0-chair_0-{method_tag}-512.json")


def build_model_args(args, method_tag, output_dir, ids_file):
    return SimpleNamespace(
        model_path=args.model_path,
        num_chunks=1,
        chunk_idx=0,
        device=args.device,
        method=method_tag,
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
        output_dir=output_dir,
        datasets="chair",
        chair_seeds="0",
        chair_samples=100,
        chair_ids_file=ids_file,
    )


def main():
    args = parse_args()
    method_tag = METHOD_TAGS[args.method]
    dev_ids = load_ids(args.dev_ids)
    validation_ids = load_ids(args.validation_ids)
    overlap = set(dev_ids) & set(validation_ids)
    if overlap:
        raise ValueError(f"Dev/validation ID overlap: {sorted(overlap)[:10]}")

    if args.smoke:
        import tempfile

        smoke_ids = dev_ids[:2]
        test_args = build_model_args(args, method_tag, "/tmp/unused", args.dev_ids)
        test = HallucinationTest(test_args)
        test.method = method_tag
        test.chair_ids = smoke_ids
        test.chair_samples = len(smoke_ids)
        test.dataset_limit = len(smoke_ids)
        runs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                test.output_dir = tmp
                test.infer_dataset("chair", 0)
                path = result_path(tmp, args.model_path, method_tag)
                runs.append(open(path).read())
        same = runs[0] == runs[1]
        rows = [json.loads(line) for line in runs[0].splitlines() if line.strip()]
        print(
            json.dumps(
                {
                    "method": args.method,
                    "method_tag": method_tag,
                    "smoke_images": len(rows),
                    "deterministic": same,
                    "sample_chars": [len(row["text"]) for row in rows],
                },
                ensure_ascii=False,
            )
        )
        return

    planned_outputs = [
        result_path(args.dev_output_dir, args.model_path, method_tag),
        result_path(args.validation_output_dir, args.model_path, method_tag),
    ]
    existing = [path for path in planned_outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: " + ", ".join(existing)
        )
    print(FROZEN_NOTE)
    print(f"Dev output: {planned_outputs[0]}")
    print(f"Validation output: {planned_outputs[1]}")

    test_args = build_model_args(args, method_tag, args.dev_output_dir, args.dev_ids)
    test = HallucinationTest(test_args)
    test.method = method_tag

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
