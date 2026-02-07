#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

repo_root = Path(__file__).resolve().parents[1]


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_capture(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd or repo_root), stderr=subprocess.STDOUT)
    except Exception:
        return None
    try:
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _git_head() -> str | None:
    return _run_capture(["git", "rev-parse", "HEAD"])


def _git_is_dirty() -> bool | None:
    try:
        unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=str(repo_root), check=False).returncode != 0
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(repo_root), check=False).returncode != 0
        return bool(unstaged or staged)
    except Exception:
        return None


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: str | Path) -> str | None:
    try:
        p = Path(path)
        return _sha256_bytes(p.read_bytes())
    except Exception:
        return None


def _sha256_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(data)


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version  # py3.8+

        return version(name)
    except Exception:
        return None


def _gather_gpu_info() -> dict[str, Any]:
    gpu: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": None,
        "nvidia_smi_list": None,
    }

    smi = _run_capture(["nvidia-smi", "-L"])
    if smi:
        gpu["nvidia_smi"] = smi
        gpu["nvidia_smi_list"] = [line.strip() for line in smi.splitlines() if line.strip()]

    # torch (optional)
    try:
        import torch  # type: ignore

        torch_info: dict[str, Any] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch_info["cuda_available"]:
            torch_info["device_count"] = int(torch.cuda.device_count())
            devices = []
            for i in range(int(torch.cuda.device_count())):
                name = None
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = None
                cap = None
                try:
                    cap = torch.cuda.get_device_capability(i)
                except Exception:
                    cap = None
                devices.append({"index": int(i), "name": name, "capability": cap})
            torch_info["devices"] = devices
        gpu["torch"] = torch_info
    except Exception:
        gpu["torch"] = None

    # onnxruntime providers (optional)
    try:
        import onnxruntime as ort  # type: ignore

        gpu["onnxruntime_providers"] = list(getattr(ort, "get_available_providers")())
        gpu["onnxruntime_version"] = getattr(ort, "__version__", None)
    except Exception:
        gpu["onnxruntime_providers"] = None
        gpu["onnxruntime_version"] = None

    return gpu


def _gather_env_info() -> dict[str, Any]:
    return {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "packages": {
            "torch": _pkg_version("torch"),
            "onnxruntime": _pkg_version("onnxruntime"),
            "tensorrt": _pkg_version("tensorrt"),
            "numpy": _pkg_version("numpy"),
            "Pillow": _pkg_version("Pillow"),
        },
    }


def _base_run_meta(*, seed: int | None, notes: str | None, config_fingerprint: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": _now_utc(),
        "seed": seed,
        "notes": notes,
        "config_hash": _sha256_json(config_fingerprint),
        "git": {"head": _git_head(), "dirty": _git_is_dirty()},
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "gpu": _gather_gpu_info(),
        "env": _gather_env_info(),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def _ensure_wrapper(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "predictions" in payload:
        preds = payload.get("predictions")
        meta = payload.get("meta")
        if isinstance(preds, list) and (meta is None or isinstance(meta, dict)):
            return {"predictions": preds, "meta": dict(meta or {})}
    if isinstance(payload, list):
        return {"predictions": payload, "meta": {}}
    if isinstance(payload, dict):
        # Legacy mapping format.
        preds = [{"image": str(k), "detections": v if isinstance(v, list) else []} for k, v in payload.items()]
        return {"predictions": preds, "meta": {}}
    raise ValueError("unsupported predictions payload")


def _subprocess_or_die(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def _parse_common_export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        choices=("dummy", "torch", "onnxrt", "trt"),
        default="dummy",
        help="Inference backend (default: dummy).",
    )
    p.add_argument("--dataset", default=None, help="YOLO-format dataset root (defaults to data/coco128).")
    p.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    p.add_argument("--output", default="reports/predictions.json", help="Predictions JSON output path.")
    p.add_argument("--notes", default=None, help="Notes to store in meta.run.")
    p.add_argument("--seed", type=int, default=None, help="Optional seed to store in meta.run.")
    p.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")
    p.add_argument("--dry-run", action="store_true", help="Backend dry-run when supported (onnxrt/trt).")

    # Torch backend (rtdetr_pose adapter).
    p.add_argument("--config", default="rtdetr_pose/configs/base.json", help="Torch config path (rtdetr_pose).")
    p.add_argument("--checkpoint", default=None, help="Torch checkpoint path (optional).")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    p.add_argument("--image-size", type=int, nargs="+", default=None, help="Torch image size (one or two ints).")
    p.add_argument("--score-threshold", type=float, default=0.3, help="Torch score threshold (default: 0.3).")
    p.add_argument("--max-detections", type=int, default=50, help="Torch max detections (default: 50).")

    # ONNXRuntime/TensorRT backend (YOLO26 exporters).
    p.add_argument("--model", default=None, help="Model path (.onnx for onnxrt, .plan for trt).")
    p.add_argument("--input-name", default="images", help="Input tensor/binding name (default: images).")
    p.add_argument("--combined-output", default="output0", help="Combined output name (default: output0).")
    p.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="abs",
        help="Combined boxes scale (default: abs).",
    )
    p.add_argument("--min-score", type=float, default=0.0, help="Score threshold (default: 0.0).")
    p.add_argument("--topk", type=int, default=300, help="Top-K per image (default: 300).")


def _export_with_backend(
    args: argparse.Namespace,
    *,
    dataset_override: str | None = None,
    dataset_meta: str | None = None,
) -> Path:
    dataset = dataset_override or (args.dataset if args.dataset else str(repo_root / "data" / "coco128"))
    dataset_fp = dataset_meta or dataset
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    if out_path.exists() and not args.force:
        return out_path

    backend = str(args.backend)

    if backend in ("dummy", "torch"):
        adapter = "dummy" if backend == "dummy" else "rtdetr_pose"
        cmd = [
            sys.executable,
            "tools/export_predictions.py",
            "--adapter",
            adapter,
            "--dataset",
            str(dataset),
            "--output",
            str(out_path),
            "--wrap",
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])

        if backend == "torch":
            cmd.extend(
                [
                    "--config",
                    str(args.config),
                    "--device",
                    str(args.device),
                    "--score-threshold",
                    str(float(args.score_threshold)),
                    "--max-detections",
                    str(int(args.max_detections)),
                ]
            )
            if args.checkpoint:
                cmd.extend(["--checkpoint", str(args.checkpoint)])
            if args.image_size:
                cmd.extend(["--image-size", *[str(int(x)) for x in args.image_size]])

        _subprocess_or_die(cmd)

        payload = _ensure_wrapper(_load_json(out_path))
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "adapter": adapter,
            "config": str(args.config) if backend == "torch" else None,
            "config_sha256": _sha256_file(repo_root / str(args.config)) if backend == "torch" else None,
            "checkpoint": str(args.checkpoint) if backend == "torch" else None,
            "checkpoint_sha256": _sha256_file(args.checkpoint) if backend == "torch" and args.checkpoint else None,
            "device": str(args.device) if backend == "torch" else None,
            "image_size": list(args.image_size) if backend == "torch" and args.image_size else None,
            "score_threshold": float(args.score_threshold) if backend == "torch" else None,
            "max_detections": int(args.max_detections) if backend == "torch" else None,
        }
        payload["meta"]["run"] = _base_run_meta(seed=args.seed, notes=args.notes, config_fingerprint=config_fp)
        _write_json(out_path, payload)
        return out_path

    if backend == "onnxrt":
        model = args.model
        if not model:
            raise SystemExit("--model is required for --backend onnxrt")
        cmd = [
            sys.executable,
            "tools/export_predictions_onnxrt.py",
            "--dataset",
            str(dataset),
            "--onnx",
            str(model),
            "--input-name",
            str(args.input_name),
            "--combined-output",
            str(args.combined_output),
            "--boxes-scale",
            str(args.boxes_scale),
            "--min-score",
            str(float(args.min_score)),
            "--topk",
            str(int(args.topk)),
            "--output",
            str(out_path),
            "--wrap",
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.dry_run:
            cmd.append("--dry-run")
        _subprocess_or_die(cmd)

        payload = _ensure_wrapper(_load_json(out_path))
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "model": str(model),
            "model_sha256": _sha256_file(model),
            "input_name": str(args.input_name),
            "combined_output": str(args.combined_output),
            "boxes_scale": str(args.boxes_scale),
            "min_score": float(args.min_score),
            "topk": int(args.topk),
            "dry_run": bool(args.dry_run),
        }
        payload["meta"]["run"] = _base_run_meta(seed=args.seed, notes=args.notes, config_fingerprint=config_fp)
        _write_json(out_path, payload)
        return out_path

    if backend == "trt":
        model = args.model
        if not model:
            raise SystemExit("--model is required for --backend trt")
        cmd = [
            sys.executable,
            "tools/export_predictions_trt.py",
            "--dataset",
            str(dataset),
            "--engine",
            str(model),
            "--input-name",
            str(args.input_name),
            "--combined-output",
            str(args.combined_output),
            "--boxes-scale",
            str(args.boxes_scale),
            "--min-score",
            str(float(args.min_score)),
            "--topk",
            str(int(args.topk)),
            "--output",
            str(out_path),
            "--wrap",
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.dry_run:
            cmd.append("--dry-run")
        _subprocess_or_die(cmd)

        payload = _ensure_wrapper(_load_json(out_path))
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "engine": str(model),
            "engine_sha256": _sha256_file(model),
            "input_name": str(args.input_name),
            "combined_output": str(args.combined_output),
            "boxes_scale": str(args.boxes_scale),
            "min_score": float(args.min_score),
            "topk": int(args.topk),
            "dry_run": bool(args.dry_run),
        }
        payload["meta"]["run"] = _base_run_meta(seed=args.seed, notes=args.notes, config_fingerprint=config_fp)
        _write_json(out_path, payload)
        return out_path

    raise SystemExit(f"unknown backend: {backend}")


def _doctor(args: argparse.Namespace) -> int:
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    report: dict[str, Any] = {
        "timestamp": _now_utc(),
        "git": {"head": _git_head(), "dirty": _git_is_dirty()},
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "gpu": _gather_gpu_info(),
        "env": _gather_env_info(),
        "tools": {
            "nvidia_smi": bool(_run_capture(["nvidia-smi", "-L"])),
            "trtexec": bool(_run_capture(["trtexec", "--version"])),
        },
    }

    warnings: list[str] = []
    if report["tools"]["nvidia_smi"] is False:
        warnings.append("nvidia-smi not found (expected on Linux+NVIDIA)")
    if report["tools"]["trtexec"] is False:
        warnings.append("trtexec not found (TensorRT engine build requires it)")
    report["warnings"] = warnings

    _write_json(out_path, report)
    print(out_path)
    return 0


def _iter_images(input_dir: Path, *, patterns: Iterable[str]) -> list[Path]:
    images: list[Path] = []
    for pat in patterns:
        images.extend(sorted(input_dir.glob(pat)))
    # De-dup while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for p in images:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _render_overlays(
    payload: dict[str, Any],
    *,
    overlays_dir: Path,
    max_images: int | None,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Pillow is required for overlays: {exc}") from exc

    overlays_dir.mkdir(parents=True, exist_ok=True)

    preds = payload.get("predictions")
    if not isinstance(preds, list):
        raise SystemExit("invalid predictions payload: missing predictions[]")

    written = 0
    index: list[dict[str, Any]] = []

    for entry in preds:
        if max_images is not None and written >= int(max_images):
            break
        if not isinstance(entry, dict):
            continue
        image_path = entry.get("image")
        if not isinstance(image_path, str) or not image_path:
            continue

        dets = entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        draw = ImageDraw.Draw(img)
        w, h = img.size
        for det in dets:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox")
            if not isinstance(bbox, dict):
                continue
            try:
                cx = float(bbox.get("cx"))
                cy = float(bbox.get("cy"))
                bw = float(bbox.get("w"))
                bh = float(bbox.get("h"))
            except Exception:
                continue
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        out_name = f"{written:06d}_{Path(image_path).name}"
        out_path = overlays_dir / out_name
        img.save(out_path)
        index.append(
            {
                "image": image_path,
                "overlay": str(out_path),
                "detections": int(len(dets)),
            }
        )
        written += 1

    return {"overlays_dir": str(overlays_dir), "count": int(written), "items": index}


def _write_html_report(
    *,
    html_path: Path,
    overlays_index: dict[str, Any],
    title: str,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    items = overlays_index.get("items") if isinstance(overlays_index, dict) else None
    if not isinstance(items, list):
        items = []

    # Use relative paths for portability.
    def rel(p: str) -> str:
        try:
            return str(Path(p).relative_to(html_path.parent))
        except Exception:
            return str(p)

    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8" />',
        f"  <title>{title}</title>",
        "  <style>",
        "    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px;}",
        "    .grid{display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:16px;}",
        "    .card{border:1px solid #ddd; border-radius:8px; padding:8px;}",
        "    img{max-width:100%; height:auto; border-radius:6px;}",
        "    .meta{color:#666; font-size:12px; overflow-wrap:anywhere;}",
        "  </style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p class='meta'>Generated: {_now_utc()}</p>",
        "<div class='grid'>",
    ]

    for it in items:
        if not isinstance(it, dict):
            continue
        overlay = it.get("overlay")
        image = it.get("image")
        dets = it.get("detections")
        if not isinstance(overlay, str) or not overlay:
            continue
        lines.extend(
            [
                "<div class='card'>",
                f"  <img src='{rel(overlay)}' />",
                f"  <div class='meta'>image: {image}</div>",
                f"  <div class='meta'>detections: {dets}</div>",
                "</div>",
            ]
        )

    lines.extend(["</div>", "</body>", "</html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")


def _predict_images(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir
    if not input_dir.exists():
        raise SystemExit(f"input dir not found: {input_dir}")

    patterns = args.glob if args.glob else ["*.jpg", "*.jpeg", "*.png"]
    images = _iter_images(input_dir, patterns=patterns)
    if args.max_images is not None:
        images = images[: int(args.max_images)]
    if not images:
        raise SystemExit(f"no images matched under: {input_dir}")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    overlays_dir = Path(args.overlays_dir)
    if not overlays_dir.is_absolute():
        overlays_dir = repo_root / overlays_dir

    html_path = None
    if args.html:
        html_path = Path(args.html)
        if not html_path.is_absolute():
            html_path = repo_root / html_path

    with tempfile.TemporaryDirectory(prefix="yolozu_predict_images_") as td:
        tmp_root = Path(td)
        split = "train2017"
        images_dir = tmp_root / "images" / split
        labels_dir = tmp_root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        mapping: dict[str, str] = {}
        for idx, src in enumerate(images):
            dst = images_dir / f"{idx:06d}_{src.name}"
            try:
                os.symlink(str(src.resolve()), str(dst))
            except Exception:
                # Fallback to copy if symlinks are not permitted.
                dst.write_bytes(src.read_bytes())
            mapping[str(dst)] = str(src.resolve())

        export_args = argparse.Namespace(**vars(args))
        export_args.dataset = str(tmp_root)
        export_args.split = split
        export_args.output = str(out_path)
        export_path = _export_with_backend(
            export_args,
            dataset_override=str(tmp_root),
            dataset_meta=str(input_dir),
        )

        payload = _ensure_wrapper(_load_json(export_path))
        # Rewrite image paths back to the original source paths for portability.
        for entry in payload.get("predictions", []):
            if not isinstance(entry, dict):
                continue
            img = entry.get("image")
            if isinstance(img, str) and img in mapping:
                entry["image"] = mapping[img]
        _write_json(out_path, payload)

    overlays_index = _render_overlays(payload, overlays_dir=overlays_dir, max_images=args.max_images)
    if html_path is not None:
        _write_html_report(html_path=html_path, overlays_index=overlays_index, title=str(args.title))
        print(html_path)
    else:
        print(out_path)
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="yolozu", description="YOLOZU unified CLI (P0/P1 building blocks).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_doctor = sub.add_parser("doctor", help="Print environment diagnostics as JSON.")
    p_doctor.add_argument("--output", default="reports/doctor.json", help="Output JSON path.")
    p_doctor.set_defaults(_fn=_doctor)

    p_export = sub.add_parser("export", help="Export predictions JSON via a selected backend.")
    _parse_common_export_args(p_export)
    p_export.set_defaults(_fn=lambda a: (print(_export_with_backend(a)), 0)[1])

    p_pi = sub.add_parser("predict-images", help="Run inference on a folder of images and write overlays/HTML.")
    _parse_common_export_args(p_pi)
    p_pi.add_argument("--input-dir", required=True, help="Folder containing images.")
    p_pi.add_argument("--glob", action="append", default=None, help="Glob pattern under --input-dir (repeatable).")
    p_pi.add_argument("--overlays-dir", default="reports/overlays", help="Directory to write overlay images.")
    p_pi.add_argument("--html", default="reports/predict_images.html", help="Optional HTML report output path.")
    p_pi.add_argument("--title", default="YOLOZU predict-images report", help="HTML title.")
    p_pi.set_defaults(_fn=_predict_images)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    fn = getattr(args, "_fn", None)
    if fn is None:
        raise SystemExit("missing handler")
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
