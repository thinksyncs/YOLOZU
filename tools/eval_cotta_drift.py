#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _extract_ttt_report(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"{label}: missing meta object")
    ttt = meta.get("ttt")
    if not isinstance(ttt, dict):
        raise ValueError(f"{label}: missing meta.ttt object")
    report = ttt.get("report")
    if not isinstance(report, dict):
        raise ValueError(f"{label}: missing meta.ttt.report object")
    return report


def _flatten_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(report.get("mode") or "stream")
    if mode != "sample":
        return [report]
    per_sample = report.get("per_sample")
    if not isinstance(per_sample, list):
        return []

    out: list[dict[str, Any]] = []
    for item in per_sample:
        if not isinstance(item, dict):
            continue
        sub = item.get("report")
        if isinstance(sub, dict):
            out.append(sub)
    return out


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _guard_breach_count(warnings: list[str]) -> int:
    total = 0
    for warning in warnings:
        w = str(warning)
        if "exceeded" in w or "non_finite" in w:
            total += 1
    return total


def _summarize_run(report: dict[str, Any]) -> dict[str, Any]:
    reports = _flatten_reports(report)
    losses: list[float] = []
    final_losses: list[float] = []
    max_total_update = 0.0
    warnings: list[str] = []
    stop_count = 0

    for rep in reports:
        rep_losses = rep.get("losses")
        if isinstance(rep_losses, list):
            local_losses = [float(x) for x in rep_losses]
            losses.extend(local_losses)
            if local_losses:
                final_losses.append(float(local_losses[-1]))

        rep_warnings = rep.get("warnings")
        if isinstance(rep_warnings, list):
            warnings.extend([str(x) for x in rep_warnings])

        if bool(rep.get("stopped_early")):
            stop_count += 1

        steps = rep.get("step_metrics")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                max_total_update = max(max_total_update, _safe_float(step.get("total_update_norm"), default=0.0))

    mean_loss = float(sum(losses) / len(losses)) if losses else 0.0
    mean_final_loss = float(sum(final_losses) / len(final_losses)) if final_losses else 0.0

    return {
        "reports_count": int(len(reports)),
        "steps_count": int(len(losses)),
        "mean_loss": float(mean_loss),
        "mean_final_loss": float(mean_final_loss),
        "max_total_update_norm": float(max_total_update),
        "guard_breaches": int(_guard_breach_count(warnings)),
        "stopped_early_count": int(stop_count),
        "warnings": warnings,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    baseline = report.get("baseline") or {}
    cotta = report.get("cotta") or {}
    decision = report.get("decision") or {}
    lines = [
        "# CoTTA drift validation report",
        "",
        f"- Generated UTC: {report.get('timestamp')}",
        f"- Stabilization pass: `{decision.get('stabilization_pass')}`",
        f"- Unsafe drift detected: `{decision.get('unsafe_drift_detected')}`",
        "",
        "## Baseline vs CoTTA",
        "",
        "| Variant | Mean Loss | Mean Final Loss | Max Total Update Norm | Guard Breaches | Stopped Early |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| baseline | "
            f"{_safe_float(baseline.get('mean_loss')):.6f} | "
            f"{_safe_float(baseline.get('mean_final_loss')):.6f} | "
            f"{_safe_float(baseline.get('max_total_update_norm')):.6f} | "
            f"{int(baseline.get('guard_breaches') or 0)} | "
            f"{int(baseline.get('stopped_early_count') or 0)} |"
        ),
        (
            "| cotta | "
            f"{_safe_float(cotta.get('mean_loss')):.6f} | "
            f"{_safe_float(cotta.get('mean_final_loss')):.6f} | "
            f"{_safe_float(cotta.get('max_total_update_norm')):.6f} | "
            f"{int(cotta.get('guard_breaches') or 0)} | "
            f"{int(cotta.get('stopped_early_count') or 0)} |"
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline-vs-CoTTA TTT runs for drift/stability evidence.")
    parser.add_argument("--baseline", required=True, help="Baseline wrapped predictions JSON path.")
    parser.add_argument("--cotta", required=True, help="CoTTA wrapped predictions JSON path.")
    parser.add_argument("--output-json", default="reports/cotta_drift_report.json", help="Output JSON report path.")
    parser.add_argument("--output-md", default="reports/cotta_drift_report.md", help="Output Markdown report path.")
    parser.add_argument(
        "--stability-loss-ratio-threshold",
        type=float,
        default=1.0,
        help="Require cotta mean final loss <= baseline mean final loss * threshold (default: 1.0).",
    )
    parser.add_argument(
        "--max-safe-total-update-norm",
        type=float,
        default=5.0,
        help="Unsafe drift threshold for CoTTA max total update norm (default: 5.0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    baseline_path = _resolve(str(args.baseline))
    cotta_path = _resolve(str(args.cotta))
    baseline_payload = _load_json(baseline_path)
    cotta_payload = _load_json(cotta_path)

    baseline_report = _extract_ttt_report(baseline_payload, label="baseline")
    cotta_report = _extract_ttt_report(cotta_payload, label="cotta")

    baseline_summary = _summarize_run(baseline_report)
    cotta_summary = _summarize_run(cotta_report)

    baseline_final = _safe_float(baseline_summary.get("mean_final_loss"), default=0.0)
    cotta_final = _safe_float(cotta_summary.get("mean_final_loss"), default=0.0)
    stability_ratio_threshold = float(args.stability_loss_ratio_threshold)
    max_safe_total_update_norm = float(args.max_safe_total_update_norm)

    if baseline_final <= 0.0:
        stabilization_by_loss = cotta_final <= baseline_final
    else:
        stabilization_by_loss = cotta_final <= (baseline_final * stability_ratio_threshold)

    unsafe_drift_detected = _safe_float(cotta_summary.get("max_total_update_norm"), default=0.0) > max_safe_total_update_norm
    guard_not_worse = int(cotta_summary.get("guard_breaches") or 0) <= int(baseline_summary.get("guard_breaches") or 0)
    stabilization_pass = bool(stabilization_by_loss and guard_not_worse and not unsafe_drift_detected)

    report = {
        "schema_version": 1,
        "kind": "cotta_drift_validation",
        "timestamp": _now_utc(),
        "inputs": {
            "baseline": {
                "path": str(baseline_path),
                "sha256": _sha256(baseline_path),
            },
            "cotta": {
                "path": str(cotta_path),
                "sha256": _sha256(cotta_path),
            },
        },
        "thresholds": {
            "stability_loss_ratio_threshold": stability_ratio_threshold,
            "max_safe_total_update_norm": max_safe_total_update_norm,
        },
        "baseline": baseline_summary,
        "cotta": cotta_summary,
        "decision": {
            "stabilization_by_loss": bool(stabilization_by_loss),
            "guard_not_worse": bool(guard_not_worse),
            "unsafe_drift_detected": bool(unsafe_drift_detected),
            "stabilization_pass": bool(stabilization_pass),
        },
    }

    out_json = _resolve(str(args.output_json))
    out_md = _resolve(str(args.output_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(out_json)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
