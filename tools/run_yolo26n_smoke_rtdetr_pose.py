import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO26n smoke run using the in-repo RT-DETR (rtdetr_pose) scaffold.")
    p.add_argument("--dataset", default=str(repo_root / "data" / "coco128"), help="YOLO-format dataset root.")
    p.add_argument("--split", default="train2017", help="Dataset split (default: train2017, matches coco128 layout).")
    p.add_argument("--config", default="rtdetr_pose/configs/base.json", help="RT-DETR config path.")
    p.add_argument("--device", default="cpu", help="Torch device for training/export_predictions (default: cpu).")
    p.add_argument("--image-size", type=int, default=64, help="Training image size (default: 64).")
    p.add_argument("--max-steps", type=int, default=5, help="Training steps cap (default: 5).")
    p.add_argument("--max-images", type=int, default=32, help="Max images for predictions export (default: 32).")
    p.add_argument("--run-dir", default=None, help="Run directory (default: runs/yolo26n_smoke/<utc>).")
    p.add_argument("--skip-fetch", action="store_true", help="Do not attempt to fetch coco128 if missing.")
    p.add_argument("--skip-train", action="store_true", help="Skip training and reuse existing checkpoint/model.onnx in run-dir.")
    p.add_argument("--strict", action="store_true", help="Strict predictions schema validation.")
    return p.parse_args(argv)


def _now_utc_compact() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _run(cmd: list[str]) -> None:
    print(shlex.join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    python = sys.executable
    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    run_dir = Path(args.run_dir) if args.run_dir else (repo_root / "runs" / "yolo26n_smoke" / _now_utc_compact())
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []

    # 1) Ensure coco128 exists (small, used by unit tests).
    if not args.skip_fetch:
        images_dir = dataset_root / "images" / str(args.split)
        labels_dir = dataset_root / "labels" / str(args.split)
        if not (images_dir.exists() and labels_dir.exists()):
            env = os.environ.copy()
            env["YOLOZU_INSECURE_SSL"] = "1"
            cmd = ["bash", "tools/fetch_coco128.sh"]
            commands.append(shlex.join(cmd))
            print("note: coco128 missing; fetching via tools/fetch_coco128.sh (YOLOZU_INSECURE_SSL=1)")
            subprocess.run(cmd, cwd=str(repo_root), check=True, env=env)

    checkpoint = run_dir / "checkpoint.pt"
    onnx_path = run_dir / "model.onnx"

    # 2) Train (tiny steps) and export ONNX.
    if not args.skip_train:
        cmd = [
            python,
            "rtdetr_pose/tools/train_minimal.py",
            "--config",
            str(args.config),
            "--dataset-root",
            str(dataset_root),
            "--split",
            str(args.split),
            "--device",
            str(args.device),
            "--real-images",
            "--image-size",
            str(int(args.image_size)),
            "--max-steps",
            str(int(args.max_steps)),
            "--run-dir",
            str(run_dir),
        ]
        commands.append(shlex.join(cmd))
        _run(cmd)

    if not checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    if not onnx_path.exists():
        raise SystemExit(f"missing onnx: {onnx_path}")

    # 3) Export predictions JSON (wrapped).
    pred_path = run_dir / "pred_yolozu_rtdetr_pose.json"
    cmd = [
        python,
        "tools/export_predictions.py",
        "--adapter",
        "rtdetr_pose",
        "--dataset",
        str(dataset_root),
        "--split",
        str(args.split),
        "--config",
        str(args.config),
        "--checkpoint",
        str(checkpoint),
        "--device",
        str(args.device),
        "--max-images",
        str(int(args.max_images)),
        "--wrap",
        "--output",
        str(pred_path),
    ]
    if args.strict:
        cmd.append("--strict")
    commands.append(shlex.join(cmd))
    _run(cmd)

    # 4) Eval suite dry-run (schema + conversion smoke; no pycocotools required).
    suite_path = run_dir / "eval_suite_dry_run.json"
    cmd = [
        python,
        "tools/eval_suite.py",
        "--protocol",
        "yolo26",
        "--dataset",
        str(dataset_root),
        "--predictions-glob",
        str(pred_path),
        "--dry-run",
        "--output",
        str(suite_path),
        "--max-images",
        str(int(args.max_images)),
    ]
    commands.append(shlex.join(cmd))
    _run(cmd)

    run_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_dir.name,
        "dataset": str(dataset_root),
        "split": str(args.split),
        "device": str(args.device),
        "artifacts": {
            "checkpoint": str(checkpoint),
            "onnx": str(onnx_path),
            "predictions": str(pred_path),
            "eval_suite_dry_run": str(suite_path),
        },
        "commands": commands,
    }
    (run_dir / "run.json").write_text(json.dumps(run_record, indent=2, sort_keys=True))
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
