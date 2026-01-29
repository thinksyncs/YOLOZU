import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import DummyAdapter, PrecomputedAdapter, RTDETRPoseAdapter
from yolozu.dataset import build_manifest
from yolozu.runner import run_adapter


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        choices=("dummy", "precomputed", "rtdetr_pose"),
        default="dummy",
        help="Which adapter to run (default: dummy).",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Predictions JSON path for --adapter precomputed.",
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
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    dataset_root = repo_root / "data" / "coco128"
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    if args.adapter == "dummy":
        adapter = DummyAdapter()
    elif args.adapter == "precomputed":
        if not args.predictions:
            raise SystemExit("--predictions is required for --adapter precomputed")
        adapter = PrecomputedAdapter(predictions_path=args.predictions)
    else:
        adapter = RTDETRPoseAdapter(config_path=args.config, checkpoint_path=args.checkpoint)

    result = run_adapter(adapter, records)
    output_dir = repo_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scenario_run.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
