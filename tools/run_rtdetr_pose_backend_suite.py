#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root / p


def _default_shape(size: int) -> str:
    s = max(1, int(size))
    return f"1x3x{s}x{s}"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end rtdetr_pose export (PyTorch→ONNX→TRT) + backend parity/benchmark suite.",
    )
    p.add_argument("--config", default="rtdetr_pose/configs/base.json", help="rtdetr_pose config path.")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")

    p.add_argument("--device", default="cuda", help="Torch device for parity/benchmark (default: cuda).")
    p.add_argument(
        "--export-device",
        default=None,
        help="Torch device for ONNX export (default: --device).",
    )

    p.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16", help="TensorRT engine precision.")
    p.add_argument("--input-name", default="images", help="Input tensor/binding name (default: images).")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17).")
    p.add_argument(
        "--dynamic-hw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export ONNX with dynamic height/width axes (default: true).",
    )

    p.add_argument("--export-image-size", type=int, default=320, help="Dummy input size for ONNX export (default: 320).")
    p.add_argument("--suite-image-size", type=int, default=640, help="Input size for suite runs (default: 640).")
    p.add_argument(
        "--max-image-size",
        type=int,
        default=960,
        help="Max size for TRT dynamic profile when shapes are not provided (default: 960).",
    )

    p.add_argument("--min-shape", default=None, help="TRT min shape (e.g. 1x3x320x320). Default: derived.")
    p.add_argument("--opt-shape", default=None, help="TRT opt shape. Default: derived.")
    p.add_argument("--max-shape", default=None, help="TRT max shape. Default: derived.")
    p.add_argument("--workspace", type=int, default=4096, help="TRT workspace size in MiB (default: 4096).")
    p.add_argument("--timing-cache", default=None, help="Timing cache path (default: <run_dir>/timing.cache).")
    p.add_argument("--trtexec", default="trtexec", help="Path to trtexec (default: trtexec).")
    p.add_argument("--extra-args", default=None, help="Extra trtexec args (quoted string).")

    p.add_argument("--batch", type=int, default=1, help="Batch size (default: 1).")
    p.add_argument("--samples", type=int, default=4, help="Random samples for parity (default: 4).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    p.add_argument("--score-atol", type=float, default=1e-4, help="Score absolute tolerance (default: 1e-4).")
    p.add_argument("--bbox-atol", type=float, default=1e-4, help="BBox absolute tolerance (default: 1e-4).")
    p.add_argument("--warmup", type=int, default=20, help="Benchmark warmup iterations (default: 20).")
    p.add_argument("--iterations", type=int, default=200, help="Benchmark iterations (default: 200).")
    p.add_argument("--backends", default="torch,onnxrt,trt", help="Backends (default: torch,onnxrt,trt).")
    p.add_argument("--embed-meta", action="store_true", help="Embed export/build meta JSON into the suite report.")

    p.add_argument("--run-id", default=None, help="Optional run id (default: timestamp).")
    p.add_argument("--run-dir", default=None, help="Optional run directory (default: runs/rtdetr_pose_backend_suite/<run_id>).")
    p.add_argument("--onnx", default=None, help="ONNX output path (default: <run_dir>/model.onnx).")
    p.add_argument("--engine", default=None, help="Engine output path (default: <run_dir>/model_<precision>.plan).")
    p.add_argument("--output", default=None, help="Suite report path (default: <run_dir>/backend_suite.json).")

    p.add_argument("--skip-onnx", action="store_true", help="Skip PyTorch→ONNX export (requires --onnx to exist).")
    p.add_argument("--skip-engine", action="store_true", help="Skip ONNX→TRT engine build.")
    p.add_argument("--force", action="store_true", help="Re-run even if outputs exist (overwrites artifacts).")
    p.add_argument("--dry-run", action="store_true", help="Run sub-tools in --dry-run mode (no GPU required).")
    return p.parse_args(argv)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    run_id = args.run_id or _now_utc().replace(":", "-")
    run_dir = _resolve(str(args.run_dir)) if args.run_dir else (repo_root / "runs" / "rtdetr_pose_backend_suite" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = _resolve(str(args.config))
    if config_path is None or not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")

    checkpoint_path = _resolve(str(args.checkpoint)) if args.checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    onnx_path = _resolve(str(args.onnx)) if args.onnx else (run_dir / "model.onnx")
    engine_path = _resolve(str(args.engine)) if args.engine else (run_dir / f"model_{args.precision}.plan")
    output_path = _resolve(str(args.output)) if args.output else (run_dir / "backend_suite.json")

    onnx_meta_path = onnx_path.with_suffix(onnx_path.suffix + ".meta.json")
    engine_meta_path = engine_path.with_suffix(engine_path.suffix + ".meta.json")
    timing_cache = _resolve(str(args.timing_cache)) if args.timing_cache else (run_dir / "timing.cache")

    min_shape = str(args.min_shape) if args.min_shape else _default_shape(int(args.export_image_size))
    opt_shape = str(args.opt_shape) if args.opt_shape else _default_shape(int(args.suite_image_size))
    max_shape = str(args.max_shape) if args.max_shape else _default_shape(int(args.max_image_size))

    skip_onnx = bool(args.skip_onnx)
    skip_engine = bool(args.skip_engine)
    if not bool(args.force):
        if onnx_path.exists() and onnx_meta_path.exists():
            skip_onnx = True
        if engine_path.exists() and engine_meta_path.exists():
            skip_engine = True

    python = sys.executable
    export_device = str(args.export_device) if args.export_device is not None else str(args.device)

    export_cmd = [
        python,
        str(repo_root / "tools" / "export_trt.py"),
        "--config",
        str(config_path),
        "--device",
        export_device,
        "--image-size",
        str(int(args.export_image_size)),
        "--onnx",
        str(onnx_path),
        "--opset",
        str(int(args.opset)),
        "--input-name",
        str(args.input_name),
        "--engine",
        str(engine_path),
        "--precision",
        str(args.precision),
        "--min-shape",
        str(min_shape),
        "--opt-shape",
        str(opt_shape),
        "--max-shape",
        str(max_shape),
        "--workspace",
        str(int(args.workspace)),
        "--timing-cache",
        str(timing_cache),
        "--trtexec",
        str(args.trtexec),
    ]
    if checkpoint_path is not None:
        export_cmd.extend(["--checkpoint", str(checkpoint_path)])
    if bool(args.dynamic_hw):
        export_cmd.append("--dynamic-hw")
    if skip_onnx:
        export_cmd.append("--skip-onnx")
    if skip_engine:
        export_cmd.append("--skip-engine")
    if args.extra_args:
        export_cmd.extend(["--extra-args", str(args.extra_args)])
    if args.dry_run:
        export_cmd.append("--dry-run")

    suite_cmd = [
        python,
        str(repo_root / "tools" / "rtdetr_pose_backend_suite.py"),
        "--config",
        str(config_path),
        "--device",
        str(args.device),
        "--image-size",
        str(int(args.suite_image_size)),
        "--batch",
        str(int(args.batch)),
        "--samples",
        str(int(args.samples)),
        "--seed",
        str(int(args.seed)),
        "--onnx",
        str(onnx_path),
        "--engine",
        str(engine_path),
        "--input-name",
        str(args.input_name),
        "--onnx-meta",
        str(onnx_meta_path),
        "--engine-meta",
        str(engine_meta_path),
        "--trtexec",
        str(args.trtexec),
        "--backends",
        str(args.backends),
        "--score-atol",
        str(float(args.score_atol)),
        "--bbox-atol",
        str(float(args.bbox_atol)),
        "--warmup",
        str(int(args.warmup)),
        "--iterations",
        str(int(args.iterations)),
        "--output",
        str(output_path),
    ]
    if checkpoint_path is not None:
        suite_cmd.extend(["--checkpoint", str(checkpoint_path)])
    if args.embed_meta:
        suite_cmd.append("--embed-meta")
    if args.dry_run:
        suite_cmd.append("--dry-run")

    _run(export_cmd)
    _run(suite_cmd)

    pipeline: dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "commands": {
            "export_trt": shlex.join(export_cmd),
            "backend_suite": shlex.join(suite_cmd),
        },
        "artifacts": {
            "onnx": str(onnx_path),
            "onnx_meta": str(onnx_meta_path),
            "engine": str(engine_path),
            "engine_meta": str(engine_meta_path),
            "suite_report": str(output_path),
        },
    }
    (run_dir / "pipeline.json").write_text(json.dumps(pipeline, indent=2, sort_keys=True), encoding="utf-8")

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
