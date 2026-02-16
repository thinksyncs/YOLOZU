from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def _safe_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
    except Exception:
        return None
    version = getattr(mod, "__version__", None)
    if version is None:
        return None
    # Some libraries (notably PyTorch) use a custom string subclass for __version__.
    # Cast to a plain str so it remains safe/portable when saved in torch checkpoints.
    return str(version)


def host_info() -> dict[str, Any]:
    """Best-effort host metadata (never raises)."""

    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = None
    try:
        fqdn = socket.getfqdn()
    except Exception:
        fqdn = None
    try:
        node = platform.node()
    except Exception:
        node = None

    return {
        "hostname": hostname,
        "fqdn": fqdn,
        "node": node,
        "pid": int(os.getpid()),
    }


def accelerator_info() -> dict[str, Any]:
    """Best-effort accelerator metadata (cuda/mps), safe without torch."""

    try:
        import torch  # type: ignore
    except Exception:
        return {"torch_available": False, "cuda": {"available": False}, "mps": {"available": False}}

    cuda_available = False
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cudnn_version = None
    try:
        cudnn_version = int(torch.backends.cudnn.version()) if hasattr(torch.backends, "cudnn") else None
    except Exception:
        cudnn_version = None

    device_count = 0
    try:
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception:
        device_count = 0

    devices: list[dict[str, Any]] = []
    if cuda_available and device_count > 0:
        for idx in range(device_count):
            name = None
            total_mem = None
            capability = None
            try:
                name = str(torch.cuda.get_device_name(idx))
            except Exception:
                name = None
            try:
                props = torch.cuda.get_device_properties(idx)
                total_mem = int(getattr(props, "total_memory", 0) or 0)
                major = getattr(props, "major", None)
                minor = getattr(props, "minor", None)
                if major is not None and minor is not None:
                    capability = [int(major), int(minor)]
            except Exception:
                total_mem = None
                capability = None
            devices.append(
                {
                    "index": int(idx),
                    "name": name,
                    "total_memory_bytes": total_mem,
                    "capability": capability,
                }
            )

    mps_available = False
    try:
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        mps_available = False

    return {
        "torch_available": True,
        "cuda": {
            "available": bool(cuda_available),
            "version": str(cuda_version) if cuda_version is not None else None,
            "cudnn": cudnn_version,
            "device_count": int(device_count),
            "devices": devices,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "mps": {"available": bool(mps_available)},
    }


def git_info(repo_root: str | Path) -> dict[str, Any]:
    """Return best-effort git metadata for repo_root.

    Never raises; returns empty dict if git is unavailable or repo_root is not a git repo.
    """

    root = Path(repo_root)
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.call(
            ["git", "-C", str(root), "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # diff --quiet returns 1 when there are changes
        is_dirty = bool(dirty != 0)
        return {"sha": sha, "dirty": is_dirty}
    except Exception:
        return {}


def versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _safe_version("numpy"),
        "torch": _safe_version("torch"),
        "Pillow": _safe_version("PIL"),
        "PyYAML": _safe_version("yaml"),
    }


def build_run_record(
    *,
    repo_root: str | Path,
    argv: list[str] | None = None,
    args: dict[str, Any] | None = None,
    dataset_root: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducibility record for logs/checkpoints.

    This is intentionally best-effort (never raises) so it can run in CI and non-git envs.
    """

    record: dict[str, Any] = {
        "host": host_info(),
        "versions": versions(),
        "accelerator": accelerator_info(),
        "git": git_info(repo_root),
    }

    if argv is None:
        argv = sys.argv[1:]
    record["argv"] = list(argv)

    if args is not None:
        record["args"] = dict(args)

    if dataset_root is not None:
        record["dataset_root"] = str(dataset_root)

    if extra:
        record["extra"] = dict(extra)

    return record
