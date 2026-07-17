import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import is_binary_question


METHOD = "rrscvhr-h18-a05-b01-official"


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def write_routed(source, destination, expected_route):
    if os.path.exists(destination):
        raise FileExistsError(f"Refusing to overwrite {destination}")
    rows = read_jsonl(source)
    routed = []
    for row in rows:
        route = "vhr" if is_binary_question(row["question"]) else "scvhr"
        if route != expected_route:
            raise ValueError(
                f"Unexpected route in {source}, question_id={row['question_id']}: {route}"
            )
        output_row = dict(row)
        output_row["response_route"] = route
        routed.append(output_row)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w") as output:
        for row in routed:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Materialize exact RR-SC-VHR outputs from validated SC-CHAIR and VHR-POPE branches."
    )
    parser.add_argument("--sc_dir", default="scvhr_official_results/instructblip-vicuna-7b")
    parser.add_argument("--vhr_dir", default="results/instructblip-vicuna-7b")
    parser.add_argument("--output_dir", default="rrscvhr_official_results/instructblip-vicuna-7b")
    args = parser.parse_args()

    manifest = []
    for seed in range(5):
        source = os.path.join(
            args.sc_dir,
            f"chair_{seed}-scvhr-h18-a05-b01-official-512.json",
        )
        destination = os.path.join(
            args.output_dir,
            f"chair_{seed}-{METHOD}-512.json",
        )
        count = write_routed(source, destination, "scvhr")
        manifest.append({"dataset": "chair", "seed": seed, "rows": count, "source": source, "output": destination})

    for split in ("popular", "adversarial", "random"):
        source = os.path.join(args.vhr_dir, f"pope_{split}_0-vhr-h18-512.json")
        destination = os.path.join(
            args.output_dir,
            f"pope_{split}_0-{METHOD}-512.json",
        )
        count = write_routed(source, destination, "vhr")
        manifest.append({"dataset": f"pope_{split}", "seed": 0, "rows": count, "source": source, "output": destination})

    manifest_path = os.path.join(args.output_dir, "materialization_manifest.json")
    with open(manifest_path, "w") as output:
        json.dump(
            {
                "method": METHOD,
                "equivalence_basis": "same-process exact smoke: SC->RR CHAIR 1/1; VHR->RR POPE 30/30",
                "files": manifest,
            },
            output,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps({"manifest": manifest_path, "files": len(manifest), "rows": sum(x["rows"] for x in manifest)}))


if __name__ == "__main__":
    main()
