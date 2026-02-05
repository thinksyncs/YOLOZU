import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import DummyAdapter, RTDETRPoseAdapter
from yolozu.dataset import build_manifest
from yolozu.predictions_transform import apply_tta


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        choices=("dummy", "rtdetr_pose"),
        default="dummy",
        help="Which adapter to run (default: dummy).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="YOLO-format dataset root (defaults to data/coco128).",
    )
    parser.add_argument(
        "--config",
        default="rtdetr_pose/configs/base.json",
        help="Config path for rtdetr_pose adapter.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for rtdetr_pose adapter (default: cpu).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        help="Image size for rtdetr_pose adapter (one value or two values).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Score threshold for rtdetr_pose adapter (default: 0.3).",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=50,
        help="Max detections per image for rtdetr_pose adapter (default: 50).",
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
    parser.add_argument("--tta", action="store_true", help="Enable TTA post-transform on predictions.")
    parser.add_argument("--tta-seed", type=int, default=None, help="Seed for TTA randomness.")
    parser.add_argument("--tta-flip-prob", type=float, default=0.5, help="Flip probability for TTA.")
    parser.add_argument("--tta-norm-only", action="store_true", help="Update only normalized bbox values for TTA.")
    parser.add_argument("--tta-log-out", default=None, help="Optional path to write TTA log JSON.")
    return parser.parse_args(argv)


def _summarize_tta(predictions, *, warnings):
    total = 0
    applied = 0
    for entry in predictions:
        mask = entry.get("tta_mask") if isinstance(entry, dict) else None
        if isinstance(mask, list):
            total += len(mask)
            applied += sum(1 for flag in mask if flag)
    ratio = float(applied) / float(total) if total else 0.0
    return {
        "detections": int(total),
        "applied": int(applied),
        "applied_ratio": float(ratio),
        "warnings": list(warnings),
    }


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = Path(args.dataset) if args.dataset else (repo_root / "data" / "coco128")
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    if args.adapter == "dummy":
        adapter = DummyAdapter()
    else:
        image_size = None
        if args.image_size:
            if len(args.image_size) == 1:
                image_size = (args.image_size[0], args.image_size[0])
            elif len(args.image_size) == 2:
                image_size = (args.image_size[0], args.image_size[1])
            else:
                raise SystemExit("--image-size expects 1 or 2 integers")
        adapter = RTDETRPoseAdapter(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            image_size=image_size or (320, 320),
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
        )

    predictions = adapter.predict(records)

    tta_warnings = []
    tta_summary = None
    if args.tta:
        tta = apply_tta(
            predictions,
            enabled=True,
            seed=args.tta_seed,
            flip_prob=args.tta_flip_prob,
            norm_only=bool(args.tta_norm_only),
        )
        predictions = tta.entries
        tta_warnings = tta.warnings
        tta_summary = _summarize_tta(predictions, warnings=tta_warnings)

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
                "tta": {
                    "enabled": bool(args.tta),
                    "seed": args.tta_seed,
                    "flip_prob": float(args.tta_flip_prob),
                    "norm_only": bool(args.tta_norm_only),
                    "warnings": tta_warnings,
                    "summary": tta_summary,
                },
            },
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    else:
        output_path.write_text(json.dumps(predictions, indent=2, sort_keys=True))

    print(output_path)

    if args.tta_log_out and args.tta:
        log_path = Path(args.tta_log_out)
        if not log_path.is_absolute():
            log_path = repo_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output": str(output_path),
            "tta": {
                "enabled": bool(args.tta),
                "seed": args.tta_seed,
                "flip_prob": float(args.tta_flip_prob),
                "norm_only": bool(args.tta_norm_only),
                "summary": tta_summary,
            },
        }
        log_path.write_text(json.dumps(log_payload, indent=2, sort_keys=True))
        print(log_path)


if __name__ == "__main__":
    main()
