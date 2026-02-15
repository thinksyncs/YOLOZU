import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.cli_args import (  # noqa: E402
    require_float_in_range,
    require_non_negative_float,
    require_non_negative_int,
    resolve_input_path,
    resolve_output_path,
)
from yolozu.dataset import build_manifest  # noqa: E402
from yolozu.pose_eval import evaluate_pose  # noqa: E402
from yolozu.predictions import load_predictions_entries  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pose fields (rot6d/log_z[/t_xyz]) against GT in dataset sidecars.")
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/ + optional sidecars).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--predictions", required=True, help="Predictions JSON (wrapped or entries list).")
    p.add_argument("--output", default="reports/pose_eval.json", help="Output JSON report path.")
    p.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching det→GT (default: 0.5).")
    p.add_argument("--min-score", type=float, default=0.0, help="Min score threshold (default: 0.0).")
    p.add_argument("--success-rot-deg", type=float, default=15.0, help="Rotation success threshold in degrees.")
    p.add_argument("--success-trans", type=float, default=0.1, help="Translation success threshold in meters.")
    p.add_argument("--keep-per-image", type=int, default=0, help="Keep up to N per-image rows in report (default: 0).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to evaluate.")
    return p.parse_args(argv)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
        keep_per_image = require_non_negative_int(args.keep_per_image, flag_name="--keep-per-image")
        iou_threshold = require_float_in_range(args.iou_threshold, flag_name="--iou-threshold", minimum=0.0, maximum=1.0)
        min_score = require_non_negative_float(args.min_score, flag_name="--min-score")
        success_rot_deg = require_non_negative_float(args.success_rot_deg, flag_name="--success-rot-deg")
        success_trans = require_non_negative_float(args.success_trans, flag_name="--success-trans")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cwd = Path.cwd()
    dataset_root = resolve_input_path(args.dataset, cwd=cwd, repo_root=repo_root)

    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest.get("images") or []
    if not isinstance(records, list):
        records = []

    if max_images is not None:
        records = records[: int(max_images)]

    pred_path = resolve_input_path(args.predictions, cwd=cwd, repo_root=repo_root)

    entries = load_predictions_entries(pred_path)
    result = evaluate_pose(
        records,
        entries,
        iou_threshold=float(iou_threshold or 0.0),
        min_score=float(min_score or 0.0),
        success_rot_deg=float(success_rot_deg or 0.0),
        success_trans=float(success_trans or 0.0),
        keep_per_image=int(keep_per_image or 0),
    )

    out_path = resolve_output_path(args.output, cwd=cwd)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": _now_utc(),
        "dataset": str(args.dataset),
        "split": manifest.get("split"),
        "predictions": str(args.predictions),
        "metrics": result.metrics,
        "counts": result.counts,
        "warnings": result.warnings,
        "per_image": result.per_image,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
