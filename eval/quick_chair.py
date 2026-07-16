import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHAIRBench


def main():
    parser = argparse.ArgumentParser(description="Fast CHAIR-only evaluation for development sweeps.")
    parser.add_argument("result_files", nargs="+")
    parser.add_argument("--ids_file", default=None)
    args = parser.parse_args()

    if args.ids_file:
        image_ids = json.load(open(args.ids_file))
    else:
        image_ids = []
        seen = set()
        for result_file in args.result_files:
            for line in open(result_file):
                image_id = int(json.loads(line)["question_id"])
                if image_id not in seen:
                    seen.add(image_id)
                    image_ids.append(image_id)
    dataset = CHAIRBench(num_samples=len(image_ids), image_ids=image_ids)
    # The upstream CHAIR implementation prints one progress update for every
    # COCO segmentation annotation. Suppress that output for fast dev loops.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _ = dataset.evaluator
    for result_file in args.result_files:
        rows = [json.loads(line) for line in open(result_file)]
        hallucinated_sentences = 0
        hallucinated_mentions = 0
        generated_mentions = 0
        lengths = []
        per_image = {}
        for row in rows:
            gt_objects, node_words = dataset.eval_hal_words(row["question_id"], row["text"])
            hallucinated = [word for word in node_words if word not in gt_objects]
            sentence_has_hallucination = bool(hallucinated)
            hallucinated_sentences += int(sentence_has_hallucination)
            hallucinated_mentions += len(hallucinated)
            generated_mentions += len(node_words)
            lengths.append(len(row["text"].split()))
            per_image[str(row["question_id"])] = {
                "CHAIRs": int(sentence_has_hallucination),
                "hallucinated_mentions": len(hallucinated),
                "generated_mentions": len(node_words),
                "hallucinated_words": hallucinated,
                "length": lengths[-1],
            }

        metrics = {
            "file": result_file,
            "num_samples": len(rows),
            "CHAIRs": 100.0 * hallucinated_sentences / max(1, len(rows)),
            "CHAIRi": 100.0 * hallucinated_mentions / max(1, generated_mentions),
            "Len": sum(lengths) / max(1, len(lengths)),
            "hallucinated_mentions": hallucinated_mentions,
            "generated_mentions": generated_mentions,
        }
        print(json.dumps(metrics, ensure_ascii=False))
        with open(result_file.replace(".json", ".quickchair.json"), "w") as f:
            json.dump({"overall": metrics, "per_image": per_image}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
