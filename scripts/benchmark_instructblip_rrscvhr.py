import argparse
import gc
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HallucinationTest


METHOD_LABELS = {
    "vhr": "vhr-h18-unified-bench",
    "scvhr": "scvhr-h18-unified-bench",
    "rrscvhr": "rrscvhr-h18-unified-bench",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Unified VHR/SC/RR latency and peak-memory benchmark.")
    parser.add_argument("--model_path", default="models/instructblip-vicuna-7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="rrscvhr_benchmark_results")
    parser.add_argument("--chair_limit", default=10, type=int)
    parser.add_argument("--pope_limit", default=100, type=int)
    return parser.parse_args()


def build_args(args):
    return SimpleNamespace(
        model_path=args.model_path,
        num_chunks=1,
        chunk_idx=0,
        device=args.device,
        method=METHOD_LABELS["vhr"],
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
        datasets="chair,pope_popular",
        chair_seeds="0",
        chair_samples=100,
        chair_ids_file="experiment/dev_chair_ids.json",
        dataset_limit=1,
    )


def measure(test, task, method_key, cycle, sample_count, root):
    method = f"{METHOD_LABELS[method_key]}-cycle{cycle}"
    test.method = method
    test.dataset_limit = sample_count
    test.output_dir = os.path.join(root, f"cycle{cycle}", task, method_key)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(test.device)
    torch.cuda.reset_peak_memory_stats(test.device)
    baseline_allocated = torch.cuda.memory_allocated(test.device)
    baseline_reserved = torch.cuda.memory_reserved(test.device)
    start = time.perf_counter()
    test.infer_dataset("chair" if task == "chair" else "pope_popular", 0)
    torch.cuda.synchronize(test.device)
    elapsed = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated(test.device)
    peak_reserved = torch.cuda.max_memory_reserved(test.device)
    return {
        "task": task,
        "method": method_key,
        "cycle": cycle,
        "samples": sample_count,
        "seconds": elapsed,
        "seconds_per_sample": elapsed / sample_count,
        "samples_per_second": sample_count / elapsed,
        "baseline_allocated_gib": baseline_allocated / 2**30,
        "baseline_reserved_gib": baseline_reserved / 2**30,
        "peak_allocated_gib": peak_allocated / 2**30,
        "peak_reserved_gib": peak_reserved / 2**30,
        "incremental_peak_allocated_gib": (peak_allocated - baseline_allocated) / 2**30,
        "incremental_peak_reserved_gib": (peak_reserved - baseline_reserved) / 2**30,
        "output_dir": test.output_dir,
        "method_label": method,
    }


def warmup(test, task, method_key, root):
    test.method = f"{METHOD_LABELS[method_key]}-warmup-{task}"
    test.dataset_limit = 1
    test.output_dir = os.path.join(root, "warmup", task, method_key)
    test.infer_dataset("chair" if task == "chair" else "pope_popular", 0)


def main():
    args = parse_args()
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Refusing to reuse benchmark output directory: {args.output_dir}")

    test = HallucinationTest(build_args(args))
    for task in ("chair", "pope"):
        for method in ("vhr", "scvhr", "rrscvhr"):
            warmup(test, task, method, args.output_dir)

    results = []
    orders = {0: ("vhr", "scvhr", "rrscvhr"), 1: ("rrscvhr", "scvhr", "vhr")}
    for cycle, order in orders.items():
        for task, count in (("chair", args.chair_limit), ("pope", args.pope_limit)):
            for method in order:
                result = measure(test, task, method, cycle, count, args.output_dir)
                results.append(result)
                print(json.dumps(result))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as output:
        json.dump(
            {
                "protocol": {
                    "model": args.model_path,
                    "device": args.device,
                    "chair_limit": args.chair_limit,
                    "pope_limit": args.pope_limit,
                    "orders": orders,
                    "peak_memory_scope": "torch.cuda allocator; model load excluded from timing",
                },
                "results": results,
            },
            output,
            indent=2,
        )
    print(json.dumps({"summary": summary_path, "measurements": len(results)}))


if __name__ == "__main__":
    main()
