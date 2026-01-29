import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections
from yolozu.dataset import build_manifest
from yolozu.predictions import load_predictions_entries, validate_predictions_entries


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="YOLO-format COCO root (images/ + labels/).")
    parser.add_argument("--split", default="val2017", help="Dataset split (default: val2017).")
    parser.add_argument(
        "--predictions-glob",
        required=True,
        help="Glob for predictions JSON files (e.g. 'reports/pred_yolo26*.json').",
    )
    parser.add_argument(
        "--bbox-format",
        choices=("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"),
        default="cxcywh_norm",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--strict", action="store_true", help="Strict prediction schema validation.")
    parser.add_argument("--output", default="reports/eval_suite.json", help="Output JSON path.")
    return parser.parse_args(argv)


def _now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = repo_root / args.dataset
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    gt, index = build_coco_ground_truth(records)
    image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt["images"]}

    pred_paths = sorted((repo_root / ".").glob(args.predictions_glob))
    if not pred_paths:
        raise SystemExit(f"no predictions matched: {args.predictions_glob}")

    results = []
    for path in pred_paths:
        entries = load_predictions_entries(path)
        validation = validate_predictions_entries(entries, strict=args.strict)
        dt = predictions_to_coco_detections(
            entries, coco_index=index, image_sizes=image_sizes, bbox_format=args.bbox_format
        )
        eval_result = evaluate_coco_map(gt, dt)
        results.append(
            {
                "name": path.stem,
                "path": str(path),
                "warnings": validation.warnings,
                **eval_result,
            }
        )

    payload = {
        "timestamp": _now_utc(),
        "dataset": str(args.dataset),
        "split": args.split,
        "bbox_format": args.bbox_format,
        "images": len(records),
        "results": results,
    }

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()

