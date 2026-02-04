import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

repo_root = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to ONNX model.")
    p.add_argument("--engine", required=True, help="Where to write TensorRT engine plan.")
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p.add_argument("--reference", required=True, help="Reference predictions JSON (PyTorch).")
    p.add_argument("--input-name", default="images", help="Input tensor name (default: images).")
    p.add_argument("--input-shape", default="1x3x640x640", help="Input shape (default: 1x3x640x640).")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 build.")
    p.add_argument("--int8", action="store_true", help="Enable INT8 build with calibration.")
    p.add_argument("--calib-dataset", default=None, help="YOLO-format dataset root for INT8 calibration.")
    p.add_argument("--calib-max-images", type=int, default=128, help="Max calibration images (default: 128).")
    p.add_argument("--calib-cache", default=None, help="Calibration cache path.")
    p.add_argument("--output", default="reports/predictions_trt.json", help="TRT predictions JSON output.")
    p.add_argument("--split", default=None, help="Dataset split under images/ and labels/.")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--boxes-output", default="boxes", help="Output name for boxes tensor.")
    p.add_argument("--scores-output", default="scores", help="Output name for scores tensor.")
    p.add_argument("--class-output", default=None, help="Optional output name for class_id tensor.")
    p.add_argument("--combined-output", default=None, help="Optional combined output name.")
    p.add_argument("--combined-format", default="xyxy_score_class", help="Combined output format.")
    p.add_argument("--raw-output", default=None, help="Optional raw output name.")
    p.add_argument("--raw-format", default="yolo_84", help="Raw output format.")
    p.add_argument("--raw-postprocess", default="native", help="Raw postprocess type.")
    p.add_argument("--boxes-format", default="xyxy", help="Boxes format (default: xyxy).")
    p.add_argument("--boxes-scale", default="norm", help="Boxes scale (default: norm).")
    p.add_argument("--min-score", type=float, default=0.001, help="Score threshold.")
    p.add_argument("--topk", type=int, default=300, help="Top-K detections per image.")
    p.add_argument("--nms-iou", type=float, default=0.7, help="IoU threshold for NMS.")
    p.add_argument("--agnostic-nms", action="store_true", help="Use class-agnostic NMS.")
    p.add_argument("--image-size", default=None, help="Fixed image size for parity check.")
    p.add_argument("--iou-thresh", type=float, default=0.99, help="IoU threshold for parity.")
    p.add_argument("--score-atol", type=float, default=1e-4, help="Score tolerance for parity.")
    p.add_argument("--bbox-atol", type=float, default=1e-4, help="BBox tolerance for parity.")
    p.add_argument("--skip-build", action="store_true", help="Skip TensorRT engine build step.")
    p.add_argument("--skip-parity", action="store_true", help="Skip parity check step.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return p.parse_args(argv)


def _append(args: list[str], flag: str, value: object | None):
    if value is None:
        return
    args.extend([flag, str(value)])


def _build_commands(args: argparse.Namespace) -> list[list[str]]:
    python = sys.executable
    cmds: list[list[str]] = []

    if not args.skip_build:
        build_cmd = [
            python,
            str(repo_root / "tools" / "build_trt_engine.py"),
            "--onnx",
            args.onnx,
            "--engine",
            args.engine,
            "--input-name",
            args.input_name,
            "--input-shape",
            args.input_shape,
        ]
        if args.fp16:
            build_cmd.append("--fp16")
        if args.int8:
            build_cmd.append("--int8")
            _append(build_cmd, "--calib-dataset", args.calib_dataset)
            _append(build_cmd, "--calib-max-images", args.calib_max_images)
            _append(build_cmd, "--calib-cache", args.calib_cache)
        cmds.append(build_cmd)

    if not args.skip_parity:
        parity_cmd = [
            python,
            str(repo_root / "tools" / "check_predictions_parity_trt.py"),
            "--reference",
            args.reference,
            "--engine",
            args.engine,
            "--dataset",
            args.dataset,
            "--input-name",
            args.input_name,
            "--boxes-output",
            args.boxes_output,
            "--scores-output",
            args.scores_output,
            "--boxes-format",
            args.boxes_format,
            "--boxes-scale",
            args.boxes_scale,
            "--min-score",
            str(args.min_score),
            "--topk",
            str(args.topk),
            "--nms-iou",
            str(args.nms_iou),
            "--output",
            args.output,
            "--iou-thresh",
            str(args.iou_thresh),
            "--score-atol",
            str(args.score_atol),
            "--bbox-atol",
            str(args.bbox_atol),
        ]
        _append(parity_cmd, "--split", args.split)
        _append(parity_cmd, "--max-images", args.max_images)
        _append(parity_cmd, "--class-output", args.class_output)
        _append(parity_cmd, "--combined-output", args.combined_output)
        _append(parity_cmd, "--combined-format", args.combined_format)
        _append(parity_cmd, "--raw-output", args.raw_output)
        _append(parity_cmd, "--raw-format", args.raw_format)
        _append(parity_cmd, "--raw-postprocess", args.raw_postprocess)
        if args.agnostic_nms:
            parity_cmd.append("--agnostic-nms")
        _append(parity_cmd, "--image-size", args.image_size)
        cmds.append(parity_cmd)

    return cmds


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cmds = _build_commands(args)
    if args.dry_run:
        for cmd in cmds:
            print(" ".join(cmd))
        return 0

    for cmd in cmds:
        subprocess.check_call(cmd, cwd=repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
