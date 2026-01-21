import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="How many images to include (default: 50).",
    )
    parser.add_argument(
        "--output",
        default="reports/predictions_dummy.json",
        help="Where to write predictions JSON.",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap output as {predictions: [...], meta: {...}} (recommended).",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=0.9,
        help="Detection score (default: 0.9).",
    )
    return parser.parse_args(argv)


def _pick_bbox_from_labels(labels):
    if labels:
        lbl = labels[0]
        return {
            "cx": float(lbl.get("cx", 0.5)),
            "cy": float(lbl.get("cy", 0.5)),
            "w": float(lbl.get("w", 0.2)),
            "h": float(lbl.get("h", 0.2)),
        }
    return {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = repo_root / "data" / "coco128"
    manifest = build_manifest(dataset_root)
    records = manifest["images"][: max(0, int(args.max_images))]

    predictions = []
    for record in records:
        bbox = _pick_bbox_from_labels(record.get("labels", []))
        detections = [
            {
                "class_id": int(record.get("labels", [{"class_id": 0}])[0].get("class_id", 0))
                if record.get("labels")
                else 0,
                "score": float(args.score),
                "bbox": bbox,
                "log_z": 0.0,
                "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "offsets": [0.0, 0.0],
                "k_delta": [0.0, 0.0, 0.0, 0.0],
            }
        ]
        predictions.append({"image": record["image"], "detections": detections})

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.wrap:
        payload = {
            "predictions": predictions,
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "generator": "make_dummy_predictions.py",
                "images": len(records),
            },
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    else:
        output_path.write_text(json.dumps(predictions, indent=2, sort_keys=True))

    print(output_path)


if __name__ == "__main__":
    main()
