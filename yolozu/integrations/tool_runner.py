from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _short_summary(ok: bool, name: str, exit_code: int) -> str:
    return f"{name}: {'ok' if ok else 'failed'} (exit={exit_code})"


def _run_yolozu_cli(name: str, args: list[str], *, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "yolozu.cli", *args]
    proc = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    ok = proc.returncode == 0
    payload: dict[str, Any] = {
        "ok": ok,
        "tool": name,
        "summary": _short_summary(ok, name, proc.returncode),
        "command": cmd,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "artifacts": {},
    }
    for key, raw_path in (artifacts or {}).items():
        path = Path(raw_path)
        payload["artifacts"][key] = str(path)
        if path.suffix.lower() == ".json" and path.exists():
            try:
                payload[f"{key}_json"] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return payload


def doctor(*, output: str = "reports/doctor.json") -> dict[str, Any]:
    return _run_yolozu_cli("doctor", ["doctor", "--output", output], artifacts={"doctor": output})


def validate_predictions(path: str, *, strict: bool = True) -> dict[str, Any]:
    args = ["validate", "predictions", path]
    if strict:
        args.append("--strict")
    return _run_yolozu_cli("validate_predictions", args)


def validate_dataset(
    dataset: str,
    *,
    split: str | None = None,
    strict: bool = True,
    mode: str = "fail",
) -> dict[str, Any]:
    args = ["validate", "dataset", dataset, "--mode", mode]
    if split:
        args.extend(["--split", split])
    if strict:
        args.append("--strict")
    return _run_yolozu_cli("validate_dataset", args)


def eval_coco(
    dataset: str,
    predictions: str,
    *,
    split: str | None = None,
    dry_run: bool = True,
    output: str = "reports/mcp_coco_eval.json",
    max_images: int | None = None,
) -> dict[str, Any]:
    args = [
        "eval-coco",
        "--dataset",
        dataset,
        "--predictions",
        predictions,
        "--output",
        output,
    ]
    if split:
        args.extend(["--split", split])
    if dry_run:
        args.append("--dry-run")
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    return _run_yolozu_cli("eval_coco", args, artifacts={"report": output})


def run_scenarios(config: str, *, extra_args: list[str] | None = None) -> dict[str, Any]:
    args = ["test", config, *(extra_args or [])]
    return _run_yolozu_cli("run_scenarios", args)


def convert_dataset(
    from_format: str,
    output: str,
    *,
    data: str | None = None,
    args_yaml: str | None = None,
    split: str | None = None,
    task: str | None = None,
    coco_root: str | None = None,
    instances_json: str | None = None,
    mode: str = "manifest",
    include_crowd: bool = False,
    force: bool = True,
) -> dict[str, Any]:
    args = ["migrate", "dataset", "--from", from_format, "--output", output, "--mode", mode]
    if data:
        args.extend(["--data", data])
    if args_yaml:
        args.extend(["--args", args_yaml])
    if split:
        args.extend(["--split", split])
    if task:
        args.extend(["--task", task])
    if coco_root:
        args.extend(["--coco-root", coco_root])
    if instances_json:
        args.extend(["--instances-json", instances_json])
    if include_crowd:
        args.append("--include-crowd")
    if force:
        args.append("--force")
    return _run_yolozu_cli("convert_dataset", args)
