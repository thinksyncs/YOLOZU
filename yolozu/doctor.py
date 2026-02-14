from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_capture(cmd: list[str], *, cwd: Path | None = None, timeout_s: float = 5.0) -> str | None:
    try:
        out = subprocess.check_output(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.STDOUT,
            timeout=float(timeout_s),
        )
    except Exception:
        return None
    try:
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version  # py3.8+

        v = version(name)
        return str(v) if v else None
    except Exception:
        return None


def _gather_git_info(*, cwd: Path) -> dict[str, Any]:
    head = _run_capture(["git", "rev-parse", "HEAD"], cwd=cwd)
    dirty = None
    try:
        unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=str(cwd), check=False).returncode != 0
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(cwd), check=False).returncode != 0
        dirty = bool(unstaged or staged)
    except Exception:
        dirty = None
    return {"head": head, "dirty": dirty}


def _gather_gpu_info() -> dict[str, Any]:
    gpu: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_list": None,
        "torch": None,
        "onnxruntime": None,
    }

    smi = _run_capture(["nvidia-smi", "-L"])
    if smi:
        gpu["nvidia_smi_list"] = [line.strip() for line in smi.splitlines() if line.strip()]

    try:
        import torch  # type: ignore

        torch_info: dict[str, Any] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch_info["cuda_available"]:
            torch_info["device_count"] = int(torch.cuda.device_count())
        gpu["torch"] = torch_info
    except Exception:
        gpu["torch"] = None

    try:
        import onnxruntime as ort  # type: ignore

        gpu["onnxruntime"] = {
            "version": getattr(ort, "__version__", None),
            "providers": list(getattr(ort, "get_available_providers")()),
        }
    except Exception:
        gpu["onnxruntime"] = None

    return gpu


def _gather_required_runtime() -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {}
    errors: list[str] = []

    def _check_import(name: str, *, import_name: str | None = None, version_attr: str = "__version__") -> None:
        mod_name = import_name or name
        try:
            mod = __import__(mod_name)
            checks[name] = {"available": True, "version": getattr(mod, version_attr, None)}
        except Exception as exc:
            checks[name] = {"available": False, "version": None, "error": repr(exc)}
            errors.append(f"missing runtime dependency: {name} ({exc})")

    _check_import("numpy")
    _check_import("Pillow", import_name="PIL")
    _check_import("PyYAML", import_name="yaml")
    return checks, errors


def build_doctor_report(*, cwd: Path | None = None) -> tuple[dict[str, Any], int]:
    from yolozu import __version__

    here = Path.cwd() if cwd is None else Path(cwd)

    required, required_errors = _gather_required_runtime()

    tools = {
        "git": bool(_run_capture(["git", "--version"], cwd=here)),
        "nvidia_smi": bool(_run_capture(["nvidia-smi", "-L"])),
        "trtexec": bool(_run_capture(["trtexec", "--version"])),
    }

    report: dict[str, Any] = {
        "kind": "yolozu_doctor",
        "schema_version": 1,
        "timestamp": _now_utc(),
        "cwd": str(here),
        "yolozu": {"version": str(__version__)},
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "packages": {
            "required": required,
            "installed_versions": {
                "yolozu": _pkg_version("yolozu"),
                "numpy": _pkg_version("numpy"),
                "Pillow": _pkg_version("Pillow"),
                "PyYAML": _pkg_version("PyYAML"),
                "torch": _pkg_version("torch"),
                "onnxruntime": _pkg_version("onnxruntime"),
                "tensorrt": _pkg_version("tensorrt"),
            },
        },
        "git": _gather_git_info(cwd=here) if tools["git"] else {"head": None, "dirty": None},
        "gpu": _gather_gpu_info(),
        "tools": tools,
        "warnings": [],
        "errors": list(required_errors),
    }

    warnings: list[str] = []
    if tools["nvidia_smi"] is False:
        warnings.append("nvidia-smi not found (expected on Linux+NVIDIA; OK on CPU-only/macOS)")
    if tools["trtexec"] is False:
        warnings.append("trtexec not found (TensorRT engine build requires it)")
    report["warnings"] = warnings

    exit_code = 0 if not required_errors else 1
    return report, int(exit_code)


def write_doctor_report(*, output: str | Path, cwd: Path | None = None) -> int:
    report, exit_code = build_doctor_report(cwd=cwd)

    if str(output) == "-":
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return int(exit_code)

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(out_path))
    return int(exit_code)

