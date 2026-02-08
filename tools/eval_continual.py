#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.continual_metrics import summarize_continual_matrix  # noqa: E402
from yolozu.dataset import build_manifest  # noqa: E402
from yolozu.predictions import load_predictions_entries  # noqa: E402
from yolozu.simple_map import evaluate_map  # noqa: E402


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _subprocess_or_die(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.stderr and proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def _write_html(*, html_path: Path, report: dict[str, Any]) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    tasks = report.get("tasks") if isinstance(report.get("tasks"), list) else []
    rows = report.get("matrix") if isinstance(report.get("matrix"), list) else []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}

    def esc(s: Any) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    lines: list[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8" />',
        "  <title>Continual Eval</title>",
        "  <style>",
        "    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px;}",
        "    .meta{color:#666; font-size:12px; overflow-wrap:anywhere;}",
        "    table{border-collapse:collapse; width:100%; margin-top:12px;}",
        "    th,td{border:1px solid #ddd; padding:6px 8px; font-size:12px; text-align:right;}",
        "    th:first-child, td:first-child{text-align:left;}",
        "  </style>",
        "</head>",
        "<body>",
        "<h1>Continual learning evaluation</h1>",
        f"<div class='meta'>run_json: {esc(meta.get('run_json'))}</div>",
        f"<div class='meta'>metric: {esc(meta.get('metric'))} key={esc(meta.get('metric_key'))}</div>",
        f"<div class='meta'>timestamp: {esc(meta.get('timestamp_utc'))}</div>",
        "<h2>Summary</h2>",
        "<table>",
        "<tr><th>metric</th><th>value</th></tr>",
        f"<tr><td>avg_acc</td><td>{esc(summary.get('avg_acc'))}</td></tr>",
        f"<tr><td>forgetting</td><td>{esc(summary.get('forgetting'))}</td></tr>",
        f"<tr><td>bwt</td><td>{esc(summary.get('bwt'))}</td></tr>",
        f"<tr><td>fwt</td><td>{esc(summary.get('fwt'))}</td></tr>",
        "</table>",
        "<h2>Matrix</h2>",
        "<table>",
        "<tr><th>checkpoint</th>",
    ]
    for t in tasks:
        lines.append(f"<th>{esc(t.get('name'))}</th>")
    lines.append("</tr>")

    for row in rows:
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        vals = row.get("values") if isinstance(row.get("values"), list) else []
        lines.append(f"<tr><td>{esc(label)}</td>")
        for v in vals:
            lines.append(f"<td>{esc(v)}</td>")
        lines.append("</tr>")

    lines.extend(["</table>", "</body>", "</html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a continual learning run (detection mAP proxy + forgetting).")
    p.add_argument("--run-json", required=True, help="Path to runs/.../continual_run.json produced by train_continual.py")
    p.add_argument("--device", default="cpu", help="Torch device for export (default: cpu).")
    p.add_argument("--image-size", type=int, default=320, help="Adapter image size (square, default: 320).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for export/eval.")
    p.add_argument(
        "--metric",
        choices=("simple_map",),
        default="simple_map",
        help="Metric backend to use (default: simple_map).",
    )
    p.add_argument(
        "--metric-key",
        choices=("map50", "map50_95"),
        default="map50_95",
        help="Which metric value to use for CL summaries (default: map50_95).",
    )
    p.add_argument("--output", default=None, help="Output JSON path (default: <run_dir>/continual_eval.json).")
    p.add_argument("--html", default=None, help="Optional HTML report path (default: <run_dir>/continual_eval.html).")
    p.add_argument("--force", action="store_true", help="Overwrite existing prediction/eval outputs.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    run_json_path = _resolve(args.run_json)
    if run_json_path is None or not run_json_path.exists():
        raise SystemExit(f"run json not found: {args.run_json}")
    run = json.loads(run_json_path.read_text(encoding="utf-8"))

    tasks = run.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("run_json.tasks must be a non-empty list")
    model_config = run.get("model_config")
    if not isinstance(model_config, str) or not model_config:
        raise SystemExit("run_json.model_config missing")
    lora_cfg = run.get("lora") if isinstance(run.get("lora"), dict) else {}
    lora_enabled = bool(lora_cfg.get("enabled", False))

    run_dir = run_json_path.parent
    out_json = _resolve(args.output) if args.output else (run_dir / "continual_eval.json")
    out_html = _resolve(args.html) if args.html else (run_dir / "continual_eval.html")
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Matrix keyed by (t,i).
    matrix_values: list[list[float | None]] = []
    matrix_rows: list[dict[str, Any]] = []

    for t, stage in enumerate(tasks):
        if not isinstance(stage, dict):
            continue
        ckpt = stage.get("checkpoint")
        label = stage.get("name") or f"task{t:02d}"
        if not isinstance(ckpt, str) or not ckpt:
            raise SystemExit(f"tasks[{t}].checkpoint missing")
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (run_dir / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise SystemExit(f"checkpoint not found: {ckpt_path}")

        row: list[float | None] = []
        row_render: list[float | None] = []

        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                row.append(None)
                row_render.append(None)
                continue
            dataset_root = task.get("dataset_root")
            val_split = task.get("val_split")
            if not isinstance(dataset_root, str) or not dataset_root:
                raise SystemExit(f"tasks[{i}].dataset_root missing")
            if not isinstance(val_split, str) or not val_split:
                raise SystemExit(f"tasks[{i}].val_split missing")

            dataset_path = Path(dataset_root)
            if not dataset_path.is_absolute():
                dataset_path = (repo_root / dataset_path).resolve()

            pred_path = eval_dir / f"pred_t{t:02d}_on_{i:02d}.json"
            if args.force or not pred_path.exists():
                cmd = [
                    sys.executable,
                    "tools/export_predictions.py",
                    "--adapter",
                    "rtdetr_pose",
                    "--dataset",
                    str(dataset_path),
                    "--split",
                    str(val_split),
                    "--config",
                    str(model_config),
                    "--checkpoint",
                    str(ckpt_path),
                    "--device",
                    str(args.device),
                    "--image-size",
                    str(int(args.image_size)),
                    "--wrap",
                    "--output",
                    str(pred_path),
                ]
                if args.max_images is not None:
                    cmd.extend(["--max-images", str(int(args.max_images))])
                if lora_enabled:
                    cmd.extend(["--lora-r", str(int(lora_cfg.get("r") or 0))])
                    if lora_cfg.get("alpha") is not None:
                        cmd.extend(["--lora-alpha", str(float(lora_cfg.get("alpha")))])
                    cmd.extend(["--lora-dropout", str(float(lora_cfg.get("dropout") or 0.0))])
                    cmd.extend(["--lora-target", str(lora_cfg.get("target") or "head")])
                    freeze_base = bool(lora_cfg.get("freeze_base", True))
                    cmd.append("--lora-freeze-base" if freeze_base else "--no-lora-freeze-base")
                    cmd.extend(["--lora-train-bias", str(lora_cfg.get("train_bias") or "none")])
                _subprocess_or_die(cmd)

            manifest = build_manifest(dataset_path, split=str(val_split))
            records = manifest["images"]
            if args.max_images is not None:
                records = records[: int(args.max_images)]
            preds = load_predictions_entries(pred_path)

            # Use a CPU-friendly proxy (no pycocotools dependency).
            r = evaluate_map(records, preds, iou_thresholds=[0.5 + 0.05 * j for j in range(10)])
            value = r.map50_95 if args.metric_key == "map50_95" else r.map50

            row.append(float(value))
            row_render.append(float(value))

        matrix_values.append(row)
        matrix_rows.append({"label": str(label), "values": row_render})

    summary = summarize_continual_matrix(matrix_values)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": _now_utc(),
        "meta": {
            "run_json": str(run_json_path),
            "metric": str(args.metric),
            "metric_key": str(args.metric_key),
            "timestamp_utc": _now_utc(),
        },
        "tasks": [{"name": str(t.get("name") or f"task{i:02d}")} for i, t in enumerate(tasks) if isinstance(t, dict)],
        "matrix": matrix_rows,
        "matrix_values": matrix_values,
        "summary": {
            "avg_acc": summary.avg_acc,
            "forgetting": summary.forgetting,
            "bwt": summary.bwt,
            "fwt": summary.fwt,
            "details": summary.details,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_html(html_path=out_html, report=payload)
    print(out_json)


if __name__ == "__main__":
    main()
