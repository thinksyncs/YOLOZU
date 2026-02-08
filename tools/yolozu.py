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
sys.path.insert(0, str(repo_root))

from yolozu.tta.presets import apply_ttt_preset_args  # noqa: E402

DEFAULT_PREDICTIONS_PATH = "reports/predictions.json"


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


def _subprocess_or_die(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.stderr and proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root / p


def _output_config_hash(path: Path) -> str | None:
    try:
        payload = _ensure_wrapper(_load_json(path))
    except Exception:
        return None
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return None
    run = meta.get("run")
    if not isinstance(run, dict):
        return None
    got = run.get("config_hash")
    return got if isinstance(got, str) and got else None


def _ensure_output_matches(path: Path, *, expected_config_hash: str) -> None:
    got = _output_config_hash(path)
    if got is None:
        raise SystemExit(f"output exists but missing meta.run.config_hash: {path} (use --force to overwrite)")
    if got != expected_config_hash:
        raise SystemExit(
            "output exists but does not match current config_hash:\n"
            f"  path: {path}\n"
            f"  expected: {expected_config_hash}\n"
            f"  got: {got}\n"
            "Use --force to overwrite, or choose a different --output/--run-dir/--cache-dir."
        )


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


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
    p.add_argument("--output", default=DEFAULT_PREDICTIONS_PATH, help="Predictions JSON output path.")
    p.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory. When set and --output is default, writes <run-dir>/predictions.json.",
    )
    p.add_argument(
        "--cache",
        action="store_true",
        help="Enable fingerprinted run cache. When set and --output is default, writes into --cache-dir/<config_hash>/predictions.json.",
    )
    p.add_argument("--cache-dir", default="runs/yolozu_runs", help="Cache root directory (default: runs/yolozu_runs).")
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
    p.add_argument("--lora-r", type=int, default=0, help="Enable LoRA by setting rank r>0 (default: 0 disables).")
    p.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha scaling (default: r).")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout on inputs (default: 0.0).")
    p.add_argument(
        "--lora-target",
        default="head",
        choices=("head", "all_linear", "all_conv1x1", "all_linear_conv1x1"),
        help="Where to apply LoRA (default: head).",
    )
    p.add_argument(
        "--lora-freeze-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze base weights and train LoRA params only (default: false).",
    )
    p.add_argument(
        "--lora-train-bias",
        choices=("none", "all"),
        default="none",
        help="If LoRA is enabled, optionally train biases too (default: none).",
    )
    p.add_argument("--tta", action="store_true", help="Enable TTA post-transform on predictions.")
    p.add_argument("--tta-seed", type=int, default=None, help="Seed for TTA randomness.")
    p.add_argument("--tta-flip-prob", type=float, default=0.5, help="Flip probability for TTA.")
    p.add_argument("--tta-norm-only", action="store_true", help="Update only normalized bbox values for TTA.")
    p.add_argument("--tta-log-out", default=None, help="Optional path to write TTA log JSON.")
    p.add_argument("--ttt", action="store_true", help="Enable test-time training (TTT) before inference.")
    p.add_argument(
        "--ttt-preset",
        choices=("safe", "adapter_only", "mim_safe"),
        default=None,
        help="Recommended TTT presets that override core knobs and fill safety guards unless explicitly set.",
    )
    p.add_argument("--ttt-method", choices=("tent", "mim"), default="tent", help="TTT method (default: tent).")
    p.add_argument(
        "--ttt-reset",
        choices=("stream", "sample"),
        default="stream",
        help="TTT reset policy: stream keeps adapted weights; sample resets per image (default: stream).",
    )
    p.add_argument("--ttt-steps", type=int, default=1, help="Total TTT steps to run (default: 1).")
    p.add_argument("--ttt-batch-size", type=int, default=1, help="TTT batch size (default: 1).")
    p.add_argument("--ttt-lr", type=float, default=1e-4, help="TTT learning rate (default: 1e-4).")
    p.add_argument(
        "--ttt-stop-on-non-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop TTT if loss/grad/update norms become non-finite (default: true).",
    )
    p.add_argument(
        "--ttt-rollback-on-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rollback last TTT step when a guard triggers (default: true).",
    )
    p.add_argument("--ttt-max-grad-norm", type=float, default=None, help="Optional grad clipping norm (default: none).")
    p.add_argument(
        "--ttt-max-update-norm",
        type=float,
        default=None,
        help="Stop if per-step weight update L2 norm exceeds this (default: none).",
    )
    p.add_argument(
        "--ttt-max-total-update-norm",
        type=float,
        default=None,
        help="Stop if total drift from initial weights exceeds this (default: none).",
    )
    p.add_argument(
        "--ttt-max-loss-ratio",
        type=float,
        default=None,
        help="Stop if loss exceeds (initial_loss * ratio) (default: none).",
    )
    p.add_argument(
        "--ttt-max-loss-increase",
        type=float,
        default=None,
        help="Stop if loss exceeds (initial_loss + delta) (default: none).",
    )
    p.add_argument(
        "--ttt-update-filter",
        choices=("all", "norm_only", "adapter_only"),
        default="all",
        help="Which parameters to update during TTT (default: all).",
    )
    p.add_argument(
        "--ttt-include",
        action="append",
        default=None,
        help="Only update parameters whose name contains this substring (repeatable).",
    )
    p.add_argument(
        "--ttt-exclude",
        action="append",
        default=None,
        help="Exclude parameters whose name contains this substring (repeatable).",
    )
    p.add_argument("--ttt-max-batches", type=int, default=1, help="Cap number of distinct batches used for TTT.")
    p.add_argument("--ttt-seed", type=int, default=None, help="Optional RNG seed for TTT.")
    p.add_argument("--ttt-mask-prob", type=float, default=0.6, help="MIM mask probability (default: 0.6).")
    p.add_argument("--ttt-patch-size", type=int, default=16, help="MIM patch size (default: 16).")
    p.add_argument("--ttt-mask-value", type=float, default=0.0, help="MIM mask fill value (default: 0.0).")
    p.add_argument("--ttt-log-out", default=None, help="Optional path to write TTT log JSON.")

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

    backend = str(args.backend)

    adapter = None
    config_fp: dict[str, Any]

    if backend in ("dummy", "torch"):
        adapter = "dummy" if backend == "dummy" else "rtdetr_pose"
        lora_enabled = bool(backend == "torch" and int(args.lora_r) > 0)
        tta_enabled = bool(args.tta)
        ttt_enabled = bool(args.ttt)
        if ttt_enabled and args.ttt_preset:
            apply_ttt_preset_args(args)
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
            "lora": {
                "enabled": lora_enabled,
                "r": int(args.lora_r) if lora_enabled else 0,
                "alpha": float(args.lora_alpha) if lora_enabled and args.lora_alpha is not None else None,
                "dropout": float(args.lora_dropout) if lora_enabled else None,
                "target": str(args.lora_target) if lora_enabled else None,
                "freeze_base": bool(args.lora_freeze_base) if lora_enabled else None,
                "train_bias": str(args.lora_train_bias) if lora_enabled else None,
            },
            "tta": {
                "enabled": tta_enabled,
                "seed": args.tta_seed if tta_enabled else None,
                "flip_prob": float(args.tta_flip_prob) if tta_enabled else None,
                "norm_only": bool(args.tta_norm_only) if tta_enabled else None,
            },
            "ttt": {
                "enabled": ttt_enabled,
                "preset": args.ttt_preset if ttt_enabled else None,
                "method": str(args.ttt_method) if ttt_enabled else None,
                "reset": str(args.ttt_reset) if ttt_enabled else None,
                "steps": int(args.ttt_steps) if ttt_enabled else None,
                "batch_size": int(args.ttt_batch_size) if ttt_enabled else None,
                "lr": float(args.ttt_lr) if ttt_enabled else None,
                "stop_on_non_finite": bool(args.ttt_stop_on_non_finite) if ttt_enabled else None,
                "rollback_on_stop": bool(args.ttt_rollback_on_stop) if ttt_enabled else None,
                "max_grad_norm": float(args.ttt_max_grad_norm) if ttt_enabled and args.ttt_max_grad_norm is not None else None,
                "max_update_norm": float(args.ttt_max_update_norm) if ttt_enabled and args.ttt_max_update_norm is not None else None,
                "max_total_update_norm": (
                    float(args.ttt_max_total_update_norm)
                    if ttt_enabled and args.ttt_max_total_update_norm is not None
                    else None
                ),
                "max_loss_ratio": float(args.ttt_max_loss_ratio) if ttt_enabled and args.ttt_max_loss_ratio is not None else None,
                "max_loss_increase": (
                    float(args.ttt_max_loss_increase)
                    if ttt_enabled and args.ttt_max_loss_increase is not None
                    else None
                ),
                "update_filter": str(args.ttt_update_filter) if ttt_enabled else None,
                "include": list(args.ttt_include) if ttt_enabled and args.ttt_include else None,
                "exclude": list(args.ttt_exclude) if ttt_enabled and args.ttt_exclude else None,
                "max_batches": int(args.ttt_max_batches) if ttt_enabled else None,
                "seed": args.ttt_seed if ttt_enabled else None,
                "mim": {
                    "mask_prob": float(args.ttt_mask_prob) if ttt_enabled else None,
                    "patch_size": int(args.ttt_patch_size) if ttt_enabled else None,
                    "mask_value": float(args.ttt_mask_value) if ttt_enabled else None,
                },
            },
        }
    elif backend == "onnxrt":
        if args.tta or args.ttt or int(args.lora_r) > 0:
            raise SystemExit("--tta/--ttt/--lora-* are only supported for --backend dummy/torch")
        model = args.model
        if not model:
            raise SystemExit("--model is required for --backend onnxrt")
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
    elif backend == "trt":
        if args.tta or args.ttt or int(args.lora_r) > 0:
            raise SystemExit("--tta/--ttt/--lora-* are only supported for --backend dummy/torch")
        model = args.model
        if not model:
            raise SystemExit("--model is required for --backend trt")
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
    else:
        raise SystemExit(f"unknown backend: {backend}")

    config_hash = _sha256_json(config_fp)

    out_path = _resolve_path(args.output)

    run_dir = None
    if args.run_dir and args.output == DEFAULT_PREDICTIONS_PATH:
        run_dir = _resolve_path(args.run_dir)
        out_path = run_dir / "predictions.json"

    cache_out = None
    if args.cache:
        cache_out = _resolve_path(args.cache_dir) / config_hash / "predictions.json"
        if args.output == DEFAULT_PREDICTIONS_PATH and not args.run_dir:
            out_path = cache_out

    if out_path.exists() and not args.force:
        _ensure_output_matches(out_path, expected_config_hash=config_hash)
        return out_path

    if cache_out is not None and cache_out.exists() and not args.force:
        _ensure_output_matches(cache_out, expected_config_hash=config_hash)
        if cache_out != out_path:
            _copy_file(cache_out, out_path)
        return out_path

    if backend in ("dummy", "torch"):
        if adapter is None:
            raise SystemExit("internal error: missing adapter")
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

        if args.tta:
            cmd.append("--tta")
        if args.tta_seed is not None:
            cmd.extend(["--tta-seed", str(int(args.tta_seed))])
        cmd.extend(["--tta-flip-prob", str(float(args.tta_flip_prob))])
        if args.tta_norm_only:
            cmd.append("--tta-norm-only")
        if args.tta_log_out:
            cmd.extend(["--tta-log-out", str(args.tta_log_out)])

        if args.ttt:
            cmd.append("--ttt")
            if args.ttt_preset:
                cmd.extend(["--ttt-preset", str(args.ttt_preset)])
            cmd.extend(["--ttt-method", str(args.ttt_method)])
            cmd.extend(["--ttt-reset", str(args.ttt_reset)])
            cmd.extend(["--ttt-steps", str(int(args.ttt_steps))])
            cmd.extend(["--ttt-batch-size", str(int(args.ttt_batch_size))])
            cmd.extend(["--ttt-lr", str(float(args.ttt_lr))])
            cmd.append(
                "--ttt-stop-on-non-finite" if bool(args.ttt_stop_on_non_finite) else "--no-ttt-stop-on-non-finite"
            )
            cmd.append(
                "--ttt-rollback-on-stop" if bool(args.ttt_rollback_on_stop) else "--no-ttt-rollback-on-stop"
            )
            if args.ttt_max_grad_norm is not None:
                cmd.extend(["--ttt-max-grad-norm", str(float(args.ttt_max_grad_norm))])
            if args.ttt_max_update_norm is not None:
                cmd.extend(["--ttt-max-update-norm", str(float(args.ttt_max_update_norm))])
            if args.ttt_max_total_update_norm is not None:
                cmd.extend(["--ttt-max-total-update-norm", str(float(args.ttt_max_total_update_norm))])
            if args.ttt_max_loss_ratio is not None:
                cmd.extend(["--ttt-max-loss-ratio", str(float(args.ttt_max_loss_ratio))])
            if args.ttt_max_loss_increase is not None:
                cmd.extend(["--ttt-max-loss-increase", str(float(args.ttt_max_loss_increase))])
            cmd.extend(["--ttt-update-filter", str(args.ttt_update_filter)])
            if args.ttt_include:
                for inc in args.ttt_include:
                    cmd.extend(["--ttt-include", str(inc)])
            if args.ttt_exclude:
                for exc in args.ttt_exclude:
                    cmd.extend(["--ttt-exclude", str(exc)])
            cmd.extend(["--ttt-max-batches", str(int(args.ttt_max_batches))])
            if args.ttt_seed is not None:
                cmd.extend(["--ttt-seed", str(int(args.ttt_seed))])
            cmd.extend(["--ttt-mask-prob", str(float(args.ttt_mask_prob))])
            cmd.extend(["--ttt-patch-size", str(int(args.ttt_patch_size))])
            cmd.extend(["--ttt-mask-value", str(float(args.ttt_mask_value))])
            if args.ttt_log_out:
                cmd.extend(["--ttt-log-out", str(args.ttt_log_out)])

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
                    "--lora-r",
                    str(int(args.lora_r)),
                    "--lora-dropout",
                    str(float(args.lora_dropout)),
                    "--lora-target",
                    str(args.lora_target),
                    "--lora-train-bias",
                    str(args.lora_train_bias),
                ]
            )
            if args.checkpoint:
                cmd.extend(["--checkpoint", str(args.checkpoint)])
            if args.image_size:
                cmd.extend(["--image-size", *[str(int(x)) for x in args.image_size]])
            if args.lora_alpha is not None:
                cmd.extend(["--lora-alpha", str(float(args.lora_alpha))])
            cmd.append("--lora-freeze-base" if bool(args.lora_freeze_base) else "--no-lora-freeze-base")

        _subprocess_or_die(cmd)
    elif backend == "onnxrt":
        cmd = [
            sys.executable,
            "tools/export_predictions_onnxrt.py",
            "--dataset",
            str(dataset),
            "--onnx",
            str(args.model),
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
    elif backend == "trt":
        cmd = [
            sys.executable,
            "tools/export_predictions_trt.py",
            "--dataset",
            str(dataset),
            "--engine",
            str(args.model),
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
    else:  # pragma: no cover
        raise SystemExit(f"unknown backend: {backend}")

    payload = _ensure_wrapper(_load_json(out_path))
    payload["meta"]["run"] = _base_run_meta(seed=args.seed, notes=args.notes, config_fingerprint=config_fp)
    _write_json(out_path, payload)

    if cache_out is not None and cache_out != out_path:
        _copy_file(out_path, cache_out)

    meta_dir = cache_out.parent if cache_out is not None else run_dir
    if meta_dir is not None:
        _write_json(meta_dir / "run_config.json", {"config_hash": config_hash, "config_fingerprint": config_fp})

    return out_path


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


def _sweep(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "tools/hpo_sweep.py",
        "--config",
        str(args.config),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.max_runs is not None:
        cmd.extend(["--max-runs", str(int(args.max_runs))])
    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
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


def _eval_instance_seg(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "tools/eval_instance_segmentation.py",
        "--dataset",
        str(args.dataset),
        "--predictions",
        str(args.predictions),
        "--output",
        str(args.output),
        "--min-score",
        str(float(args.min_score)),
        "--diag-iou",
        str(float(args.diag_iou)),
        "--per-image-limit",
        str(int(args.per_image_limit)),
    ]

    if args.split is not None:
        cmd.extend(["--split", str(args.split)])
    if args.pred_root is not None:
        cmd.extend(["--pred-root", str(args.pred_root)])
    if args.classes is not None:
        cmd.extend(["--classes", str(args.classes)])
    if args.html is not None:
        cmd.extend(["--html", str(args.html)])
    if args.title is not None:
        cmd.extend(["--title", str(args.title)])
    if args.overlays_dir is not None:
        cmd.extend(["--overlays-dir", str(args.overlays_dir)])

    cmd.extend(["--max-overlays", str(int(args.max_overlays))])
    cmd.extend(["--overlay-sort", str(args.overlay_sort)])
    cmd.extend(["--overlay-max-size", str(int(args.overlay_max_size))])
    cmd.extend(["--overlay-alpha", str(float(args.overlay_alpha))])

    if args.max_images is not None:
        cmd.extend(["--max-images", str(int(args.max_images))])
    if args.allow_rgb_masks:
        cmd.append("--allow-rgb-masks")

    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="yolozu", description="YOLOZU unified CLI (P0/P1/P2 building blocks).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_doctor = sub.add_parser("doctor", help="Print environment diagnostics as JSON.")
    p_doctor.add_argument("--output", default="reports/doctor.json", help="Output JSON path.")
    p_doctor.set_defaults(_fn=_doctor)

    p_sweep = sub.add_parser("sweep", help="Run a parameter sweep (wrapper around tools/hpo_sweep.py).")
    p_sweep.add_argument("--config", required=True, help="Path to sweep config JSON.")
    p_sweep.add_argument("--resume", action="store_true", help="Skip runs already present in results jsonl.")
    p_sweep.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p_sweep.add_argument("--max-runs", type=int, default=None, help="Optional cap for number of runs.")
    p_sweep.set_defaults(_fn=_sweep)

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

    p_is = sub.add_parser("eval-instance-seg", help="Evaluate instance segmentation predictions (PNG masks) and write a report.")
    p_is.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p_is.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p_is.add_argument("--predictions", required=True, help="Instance segmentation predictions JSON.")
    p_is.add_argument("--pred-root", default=None, help="Optional root to resolve relative prediction mask paths.")
    p_is.add_argument("--classes", default=None, help="Optional classes.txt/classes.json for class_id→name.")
    p_is.add_argument("--output", default="reports/instance_seg_eval.json", help="Output JSON report path.")
    p_is.add_argument("--html", default=None, help="Optional HTML report path.")
    p_is.add_argument("--title", default="YOLOZU instance segmentation eval report", help="HTML title.")
    p_is.add_argument("--overlays-dir", default=None, help="Optional directory to write overlay images for HTML.")
    p_is.add_argument("--max-overlays", type=int, default=0, help="Max overlays to render (default: 0).")
    p_is.add_argument(
        "--overlay-sort",
        choices=("worst", "best", "first"),
        default="worst",
        help="How to select overlay samples (default: worst).",
    )
    p_is.add_argument("--overlay-max-size", type=int, default=768, help="Max size (max(H,W)) for overlay images (default: 768).")
    p_is.add_argument("--overlay-alpha", type=float, default=0.5, help="Mask overlay alpha (default: 0.5).")
    p_is.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold for predictions (default: 0.0).")
    p_is.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to evaluate.")
    p_is.add_argument("--diag-iou", type=float, default=0.5, help="IoU threshold used for per-image diagnostics/overlay selection (default: 0.5).")
    p_is.add_argument("--per-image-limit", type=int, default=100, help="How many per-image rows to store in the report/meta and HTML (default: 100).")
    p_is.add_argument(
        "--allow-rgb-masks",
        action="store_true",
        help="Allow 3-channel masks (uses channel 0; intended for grayscale stored as RGB).",
    )
    p_is.set_defaults(_fn=_eval_instance_seg)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    fn = getattr(args, "_fn", None)
    if fn is None:
        raise SystemExit("missing handler")
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
