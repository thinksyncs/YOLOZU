from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .core import fail_response, ok_response

_ALLOWED_TOP_LEVEL = {
    "doctor",
    "validate",
    "eval-coco",
    "test",
    "migrate",
    "train",
    "export",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _looks_like_path_token(token: str) -> bool:
    return "/" in token or token.startswith(".") or token.endswith((".json", ".yaml", ".yml", ".onnx", ".pt", ".engine"))


def _guard_path_token(token: str) -> None:
    if not _looks_like_path_token(token):
        return
    p = Path(token)
    if ".." in p.parts:
        raise ValueError(f"path traversal is not allowed: {token}")
    if p.is_absolute():
        root = repo_root().resolve()
        resolved = p.resolve()
        if root not in resolved.parents and resolved != root:
            raise ValueError(f"absolute path outside workspace is not allowed: {token}")


def run_cli_tool(name: str, args: list[str], *, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    if not args:
        return fail_response(name, message="empty command")
    if args[0] not in _ALLOWED_TOP_LEVEL:
        return fail_response(name, message=f"command not allowed: {args[0]}")

    try:
        for token in args:
            _guard_path_token(token)
    except Exception as exc:
        return fail_response(name, message=str(exc), exc=exc)

    cmd = [sys.executable, "-m", "yolozu.cli", *args]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        payload = fail_response(
            name,
            message=f"cli failed with exit code {proc.returncode}",
            exit_code=proc.returncode,
        )
        payload["command"] = cmd
        payload["stdout"] = proc.stdout
        payload["stderr"] = proc.stderr
        payload["artifacts"] = {}
        return payload

    payload = ok_response(
        name,
        exit_code=proc.returncode,
        data={
            "command": cmd,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "artifacts": {},
        },
    )
    for key, raw_path in (artifacts or {}).items():
        path = Path(raw_path)
        payload["artifacts"][key] = str(path)
        if path.suffix.lower() == ".json" and path.exists():
            try:
                payload[f"{key}_json"] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return payload
