import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to ONNX model.")
    p.add_argument("--engine", default="engines/model.plan", help="Where to write the TensorRT engine.")
    p.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16", help="Engine precision.")
    p.add_argument("--input-name", default="images", help="ONNX input tensor name.")
    p.add_argument("--min-shape", default="1x3x640x640", help="Min input shape (e.g., 1x3x640x640).")
    p.add_argument("--opt-shape", default="1x3x640x640", help="Opt input shape.")
    p.add_argument("--max-shape", default="1x3x640x640", help="Max input shape.")
    p.add_argument("--workspace", type=int, default=4096, help="Workspace size in MiB.")
    p.add_argument("--timing-cache", default="engines/timing.cache", help="Timing cache path.")
    p.add_argument("--calib-cache", default=None, help="INT8 calibration cache path (required for int8).")
    p.add_argument("--calib-dataset", default=None, help="Optional dataset root to build a calibration image list.")
    p.add_argument("--calib-split", default=None, help="Dataset split for calibration list (default: auto).")
    p.add_argument("--calib-images", type=int, default=100, help="Number of images for calibration list.")
    p.add_argument(
        "--calib-list-output",
        default="reports/calib_images.txt",
        help="Where to write calibration image list (if --calib-dataset is set).",
    )
    p.add_argument("--trtexec", default="trtexec", help="Path to trtexec binary.")
    p.add_argument("--extra-args", default=None, help="Extra trtexec args (quoted string).")
    p.add_argument("--meta-output", default="reports/trt_engine_meta.json", help="Where to write build metadata JSON.")
    p.add_argument("--dry-run", action="store_true", help="Print command and write meta without running trtexec.")
    return p.parse_args(argv)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_input_name(network: object, requested: str) -> str:
    """Best-effort input-name resolution helper.

    This is primarily for compatibility with older TensorRT-python based builders
    and for unit tests; the current implementation uses trtexec.
    """

    if not hasattr(network, "num_inputs"):
        return requested
    try:
        count = int(getattr(network, "num_inputs"))
        inputs = [getattr(network.get_input(i), "name", None) for i in range(count)]
        inputs = [str(name) for name in inputs if name]
    except Exception:
        return requested
    if requested in inputs:
        return requested
    if inputs:
        return inputs[0]
    return requested


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _write_calib_list(dataset_root: str, *, split: str | None, limit: int, output_path: Path) -> list[str]:
    manifest = build_manifest(dataset_root, split=split)
    records = manifest["images"][: max(0, int(limit))]
    paths = [str(r["image"]) for r in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(paths) + ("\n" if paths else ""))
    return paths


def _build_command(args: argparse.Namespace, *, onnx_path: Path, engine_path: Path, timing_cache: Path) -> list[str]:
    cmd = [
        args.trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={args.input_name}:{args.min_shape}",
        f"--optShapes={args.input_name}:{args.opt_shape}",
        f"--maxShapes={args.input_name}:{args.max_shape}",
        f"--workspace={int(args.workspace)}",
        f"--timingCacheFile={timing_cache}",
    ]
    if args.precision == "fp16":
        cmd.append("--fp16")
    elif args.precision == "int8":
        cmd.append("--int8")
        if args.calib_cache:
            cmd.append(f"--calib={_resolve_path(args.calib_cache)}")
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
        return out.decode("utf-8").strip() or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    onnx_path = _resolve_path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"onnx not found: {onnx_path}")

    engine_path = _resolve_path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    timing_cache = _resolve_path(args.timing_cache)
    timing_cache.parent.mkdir(parents=True, exist_ok=True)

    calib_list_path = None
    calib_list = None
    if args.calib_dataset:
        calib_list_path = _resolve_path(args.calib_list_output)
        calib_list = _write_calib_list(
            args.calib_dataset,
            split=args.calib_split,
            limit=args.calib_images,
            output_path=calib_list_path,
        )

    cmd = _build_command(args, onnx_path=onnx_path, engine_path=engine_path, timing_cache=timing_cache)

    meta: dict[str, Any] = {
        "timestamp": _now_utc(),
        "git_head": _git_head(),
        "onnx": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "engine": str(engine_path),
        "precision": args.precision,
        "input_name": args.input_name,
        "shapes": {"min": args.min_shape, "opt": args.opt_shape, "max": args.max_shape},
        "workspace_mib": int(args.workspace),
        "timing_cache": str(timing_cache),
        "calib_cache": None if not args.calib_cache else str(_resolve_path(args.calib_cache)),
        "calib_list": None if calib_list_path is None else str(calib_list_path),
        "calib_images": None if calib_list is None else len(calib_list),
        "command": cmd,
        "command_str": shlex.join(cmd),
        "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": sys.version,
    }

    meta_path = _resolve_path(args.meta_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    if args.precision == "int8" and not args.calib_cache:
        print("warning: --precision int8 without --calib-cache; trtexec may refuse to build", file=sys.stderr)

    if args.dry_run:
        print(meta["command_str"])
        return 0

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"trtexec not found: {args.trtexec}") from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
