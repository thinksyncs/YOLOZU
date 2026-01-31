import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections
from yolozu.dataset import build_manifest
from yolozu.eval_protocol import apply_eval_protocol_args, load_eval_protocol
from yolozu.predictions import load_predictions_entries, validate_predictions_entries


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=("yolo26",),
        default=None,
        help="Apply canonical evaluation protocol presets (pins split/bbox_format).",
    )
    parser.add_argument("--dataset", default="data/coco128", help="YOLO-format COCO root (images/ + labels/).")
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto (val2017 if present else train2017).",
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip COCOeval and only validate/convert predictions (no pycocotools required).",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--strict", action="store_true", help="Strict prediction schema validation.")
    parser.add_argument("--output", default="reports/eval_suite.json", help="Output JSON path.")
    return parser.parse_args(argv)


def _resolve_args(argv):
    args = _parse_args(argv)
    protocol = load_eval_protocol(args.protocol) if args.protocol else None
    if protocol:
        args = apply_eval_protocol_args(args, protocol)
    return args, protocol


def _now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv=None):
    args, protocol = _resolve_args(sys.argv[1:] if argv is None else argv)

    dataset_root = repo_root / args.dataset
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    split_effective = manifest["split"]
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
        if args.dry_run:
            eval_result = {
                "metrics": {
                    "map50_95": None,
                    "map50": None,
                    "map75": None,
                    "ar100": None,
                },
                "stats": [],
                "dry_run": True,
                "counts": {"images": len(records), "detections": len(dt)},
            }
        else:
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
        "report_schema_version": 1,
        "timestamp": _now_utc(),
        "protocol_id": args.protocol,
        "protocol": protocol,
        "dataset": str(args.dataset),
        "split": split_effective,
        "split_requested": args.split,
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
