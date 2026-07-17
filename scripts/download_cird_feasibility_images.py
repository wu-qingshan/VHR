import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download only the fixed COCO-train images used by the CIRD teacher gate."
    )
    parser.add_argument(
        "--ids_file", default="experiment/cird_teacher_feasibility_train_ids.json"
    )
    parser.add_argument(
        "--captions", default="data/coco/annotations/captions_train2014.json"
    )
    parser.add_argument("--output_dir", default="data/coco/train2014")
    parser.add_argument("--workers", default=12, type=int)
    return parser.parse_args()


def download_one(image, output_dir):
    destination = os.path.join(output_dir, image["file_name"])
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return "existing", destination
    url = f"http://images.cocodataset.org/train2014/{image['file_name']}"
    temporary = destination + ".part"
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, destination)
    return "downloaded", destination


def main():
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    with open(args.ids_file) as source:
        image_ids = [int(image_id) for image_id in json.load(source)]
    coco = COCO(args.captions)
    images = coco.loadImgs(image_ids)
    if len(images) != len(image_ids):
        raise RuntimeError("Some fixed CIRD image IDs are absent from COCO train captions")
    os.makedirs(args.output_dir, exist_ok=True)

    counts = {"downloaded": 0, "existing": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, image, args.output_dir): image["id"]
            for image in images
        }
        for future in as_completed(futures):
            status, path = future.result()
            counts[status] += 1
            print(f"{status}: {path}")
    print(json.dumps({"images": len(images), **counts}, indent=2))


if __name__ == "__main__":
    main()
