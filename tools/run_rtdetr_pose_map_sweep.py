import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]


def _now_utc_compact() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _split_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _run(cmd: list[str], *, quiet: bool = False) -> None:
    if not quiet:
        print(shlex.join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True, stdout=(subprocess.DEVNULL if quiet else None))


def _load_metrics(eval_suite_path: Path) -> dict[str, float | None]:
    payload = json.loads(eval_suite_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return {}
    metrics = results[0].get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _pred_stats(pred_path: Path) -> dict[str, float | int | None]:
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    preds = payload.get("predictions") or []
    if not isinstance(preds, list):
        preds = []

    dets_total = 0
    has_keypoints = 0
    has_log_z = 0
    has_rot6d = 0
    for entry in preds:
        dets = entry.get("detections") if isinstance(entry, dict) else None
        if not isinstance(dets, list):
            continue
        dets_total += len(dets)
        for det in dets:
            if not isinstance(det, dict):
                continue
            if det.get("keypoints") is not None:
                has_keypoints += 1
            if det.get("log_z") is not None:
                has_log_z += 1
            rot6d = det.get("rot6d")
            if isinstance(rot6d, list) and len(rot6d) == 6:
                has_rot6d += 1

    images = len(preds)
    dets_per_image = (float(dets_total) / float(images)) if images else None
    pose_log_z_frac = (float(has_log_z) / float(dets_total)) if dets_total else None
    pose_rot6d_frac = (float(has_rot6d) / float(dets_total)) if dets_total else None

    return {
        "images": images,
        "dets_total": dets_total,
        "dets_per_image": dets_per_image,
        "pose_log_z_frac": pose_log_z_frac,
        "pose_rot6d_frac": pose_rot6d_frac,
        "keypoints_dets": has_keypoints,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RTDETRPose train→predict→COCOeval sweep and report mAP trend.")
    p.add_argument("--epochs", default="1,5,20,50", help="Comma-separated epochs to sweep (default: 1,5,20,50).")
    p.add_argument("--config", default="configs/yolo26_rtdetr_pose/yolo26n.json", help="RTDETRPose config path.")
    p.add_argument("--dataset", default="data/coco128", help="YOLO-format dataset root.")
    p.add_argument("--split", default="train2017", help="Split under images/ and labels/.")
    p.add_argument("--device", default="cuda", help="Torch device for training/export_predictions.")
    p.add_argument("--image-size", type=int, default=320, help="Training/export image size.")
    p.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Training learning rate.")
    p.add_argument("--max-images", type=int, default=128, help="Max images for prediction/eval.")
    p.add_argument("--score-threshold", type=float, default=0.01, help="Prediction score threshold.")
    p.add_argument("--max-detections", type=int, default=300, help="Max detections per image before thresholding.")
    p.add_argument("--run-dir", default=None, help="Base run directory (default: runs/rtdetr_pose_map_sweep/<utc>).")
    p.add_argument("--quiet", action="store_true", help="Suppress tool stdout.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    epochs_list = _split_int_csv(str(args.epochs))
    if not epochs_list:
        raise SystemExit("--epochs must be non-empty")

    run_base = Path(args.run_dir) if args.run_dir else (repo_root / "runs" / "rtdetr_pose_map_sweep" / _now_utc_compact())
    if not run_base.is_absolute():
        run_base = repo_root / run_base
    run_base.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for epochs in epochs_list:
        run_dir = run_base / f"ep{int(epochs)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        ckpt = run_dir / "checkpoint.pt"
        onnx = run_dir / "model.onnx"
        pred = run_dir / "pred.json"
        suite = run_dir / "eval_suite.json"

        print(f"\n=== epochs={int(epochs)} run={run_dir} ===")

        _run(
            [
                sys.executable,
                "rtdetr_pose/tools/train_minimal.py",
                "--config",
                str(args.config),
                "--dataset-root",
                str(args.dataset),
                "--split",
                str(args.split),
                "--device",
                str(args.device),
                "--real-images",
                "--image-size",
                str(int(args.image_size)),
                "--batch-size",
                str(int(args.batch_size)),
                "--lr",
                str(float(args.lr)),
                "--use-matcher",
                "--epochs",
                str(int(epochs)),
                "--max-steps",
                "1000000",
                "--run-dir",
                str(run_dir),
                "--checkpoint-out",
                str(ckpt),
                "--onnx-out",
                str(onnx),
                "--log-every",
                "200",
            ],
            quiet=bool(args.quiet),
        )

        _run(
            [
                sys.executable,
                "tools/export_predictions.py",
                "--adapter",
                "rtdetr_pose",
                "--dataset",
                str(args.dataset),
                "--split",
                str(args.split),
                "--config",
                str(args.config),
                "--checkpoint",
                str(ckpt),
                "--device",
                str(args.device),
                "--image-size",
                str(int(args.image_size)),
                "--score-threshold",
                str(float(args.score_threshold)),
                "--max-detections",
                str(int(args.max_detections)),
                "--max-images",
                str(int(args.max_images)),
                "--wrap",
                "--output",
                str(pred),
            ],
            quiet=bool(args.quiet),
        )

        _run(
            [
                sys.executable,
                "tools/eval_suite.py",
                "--dataset",
                str(args.dataset),
                "--split",
                str(args.split),
                "--predictions-glob",
                str(pred),
                "--bbox-format",
                "cxcywh_norm",
                "--max-images",
                str(int(args.max_images)),
                "--output",
                str(suite),
            ],
            quiet=bool(args.quiet),
        )

        metrics = _load_metrics(suite)
        stats = _pred_stats(pred)

        row = {
            "epochs": int(epochs),
            "run_dir": str(run_dir),
            "map50_95": metrics.get("map50_95"),
            "map50": metrics.get("map50"),
            **stats,
        }
        summary.append(row)

        print(
            "map50_95",
            row["map50_95"],
            "map50",
            row["map50"],
            "dets/img",
            row["dets_per_image"],
            "pose(log_z,rot6d)%",
            row["pose_log_z_frac"],
            row["pose_rot6d_frac"],
            "keypoints_dets",
            row["keypoints_dets"],
        )

    (run_base / "summary.json").write_text(json.dumps({"runs": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nsummary: {run_base / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

