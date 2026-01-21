import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import DummyAdapter, RTDETRPoseAdapter
from yolozu.dataset import build_manifest


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        choices=("dummy", "rtdetr_pose"),
        default="dummy",
        help="Which adapter to run (default: dummy).",
    )
    parser.add_argument(
        "--config",
        default="rtdetr_pose/configs/base.json",
        help="Config path for rtdetr_pose adapter.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint for rtdetr_pose adapter.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for number of images (for quick smoke runs).",
    )
    parser.add_argument(
        "--output",
        default="reports/predictions.json",
        help="Where to write predictions JSON.",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap output as {predictions: [...], meta: {...}} (recommended).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = repo_root / "data" / "coco128"
    manifest = build_manifest(dataset_root)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    if args.adapter == "dummy":
        adapter = DummyAdapter()
    else:
        adapter = RTDETRPoseAdapter(config_path=args.config, checkpoint_path=args.checkpoint)

    predictions = adapter.predict(records)

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.wrap:
        payload = {
            "predictions": predictions,
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "adapter": args.adapter,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "images": len(records),
            },
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    else:
        output_path.write_text(json.dumps(predictions, indent=2, sort_keys=True))

    print(output_path)


if __name__ == "__main__":
    main()
