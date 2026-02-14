from __future__ import annotations

import platform
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
        "versions": versions(),
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
