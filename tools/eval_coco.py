import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections
from yolozu.dataset import build_manifest
from yolozu.predictions import load_predictions_entries


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/coco128", help="YOLO-format dataset root (images/ + labels/).")
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto.",
    )
    parser.add_argument("--predictions", required=True, help="Predictions JSON path (YOLOZU format).")
    parser.add_argument(
        "--bbox-format",
        choices=("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"),
        default="cxcywh_norm",
        help="How to interpret detection bbox fields (default: cxcywh_norm).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for number of images (for quick smoke runs).",
    )
    parser.add_argument("--output", default="reports/coco_eval.json", help="Where to write evaluation JSON.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = repo_root / args.dataset
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    gt, index = build_coco_ground_truth(records)
    image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt["images"]}

    preds = load_predictions_entries(repo_root / args.predictions)
    dt = predictions_to_coco_detections(preds, coco_index=index, image_sizes=image_sizes, bbox_format=args.bbox_format)
    result = evaluate_coco_map(gt, dt)

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(args.dataset),
        "predictions": str(args.predictions),
        "bbox_format": args.bbox_format,
        **result,
    }

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
