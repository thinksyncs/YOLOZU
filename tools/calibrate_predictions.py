import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.calibration import CalibConfig, calibrate_predictions_lbfgs
from yolozu.dataset import build_manifest
from yolozu.predictions import load_predictions_entries


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Calibrate predictions using L-BFGS (depth scale s + optional shared k_delta).")
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root.")
    p.add_argument("--predictions", required=True, help="Input predictions JSON path (any supported YOLOZU shape).")
    p.add_argument("--output", required=True, help="Output predictions JSON path.")
    p.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (e.g. val2017).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")

    p.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for det<->GT matching.")

    p.add_argument("--optimize-k-delta", action="store_true", help="Also optimize a shared k_delta (requires intrinsics + image size + t_gt).")
    p.add_argument(
        "--image-hw",
        type=float,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Override image_hw (pixels) used for u,v conversion when bbox is normalized.",
    )

    p.add_argument("--lbfgs-max-iter", type=int, default=30, help="L-BFGS max_iter.")
    p.add_argument("--lbfgs-lr", type=float, default=1.0, help="L-BFGS lr (step size).")

    p.add_argument("--w-z", type=float, default=1.0, help="Weight for z supervision (from t_gt.z).")
    p.add_argument("--w-t", type=float, default=0.2, help="Weight for translation supervision (t_xyz vs t_gt).")
    p.add_argument("--reg-k", type=float, default=1e-2, help="L2 regularization weight for shared k_delta.")
    p.add_argument("--reg-log-s", type=float, default=1e-4, help="L2 regularization weight for log(s).")

    p.add_argument("--wrap", action="store_true", help="Write wrapped output {schema_version,predictions,meta}.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = Path(args.dataset)
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    predictions = load_predictions_entries(args.predictions)

    config = CalibConfig(
        iou_threshold=float(args.iou_threshold),
        optimize_k_delta=bool(args.optimize_k_delta),
        image_hw_override=(float(args.image_hw[0]), float(args.image_hw[1])) if args.image_hw else None,
        lbfgs_max_iter=int(args.lbfgs_max_iter),
        lbfgs_lr=float(args.lbfgs_lr),
        w_z=float(args.w_z),
        w_t=float(args.w_t),
        reg_k=float(args.reg_k),
        reg_log_s=float(args.reg_log_s),
    )

    calibrated, report = calibrate_predictions_lbfgs(records, predictions, config=config)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.wrap:
        payload = {
            "schema_version": 1,
            "predictions": calibrated,
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tool": "tools/calibrate_predictions.py",
                "dataset": str(dataset_root),
                "split": manifest.get("split"),
                "calibration": {
                    "method": "lbfgs",
                    **report.to_dict(),
                    "config": {
                        "iou_threshold": float(args.iou_threshold),
                        "optimize_k_delta": bool(args.optimize_k_delta),
                        "image_hw": list(args.image_hw) if args.image_hw else None,
                        "lbfgs_max_iter": int(args.lbfgs_max_iter),
                        "lbfgs_lr": float(args.lbfgs_lr),
                        "w_z": float(args.w_z),
                        "w_t": float(args.w_t),
                        "reg_k": float(args.reg_k),
                        "reg_log_s": float(args.reg_log_s),
                    },
                },
            },
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    else:
        out_path.write_text(json.dumps(calibrated, indent=2, sort_keys=True))

    print(out_path)


if __name__ == "__main__":
    main()
