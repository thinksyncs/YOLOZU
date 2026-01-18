import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.scenario_suite import build_report


def _average_metrics(report):
    scenarios = report.get("scenarios", [])
    totals = {
        "fps": 0.0,
        "recall": 0.0,
        "depth_error": 0.0,
        "rejection_rate": 0.0,
    }
    for entry in scenarios:
        metrics = entry.get("metrics", {})
        for key in totals:
            totals[key] += float(metrics.get(key, 0.0))
    count = max(1, len(scenarios))
    return {key: value / count for key, value in totals.items()}


def _scenario_metrics(report, name):
    for entry in report.get("scenarios", []):
        if entry.get("name") == name:
            return entry.get("metrics", {})
    return {}


def main():
    baseline_path = repo_root / "reports" / "baseline.json"
    if not baseline_path.exists():
        raise SystemExit("baseline report missing; run tools/run_baseline.py first")

    baseline = json.loads(baseline_path.read_text())
    baseline_report = baseline.get("scenario_report", {})
    current_report = build_report()

    base_avg = _average_metrics(baseline_report)
    curr_avg = _average_metrics(current_report)

    jitter_on = _scenario_metrics(current_report, "handheld_jitter_on")
    jitter_off = _scenario_metrics(current_report, "handheld_jitter_off")

    checks = {
        "stage3_fp_recall_fps": curr_avg["recall"] >= base_avg["recall"] - 0.05
        and curr_avg["fps"] >= 30.0
        and curr_avg["rejection_rate"] <= base_avg["rejection_rate"] + 0.05,
        "stage4_constraints": curr_avg["depth_error"] <= base_avg["depth_error"] + 0.05
        and curr_avg["fps"] >= 30.0,
        "stage5_jitter": float(jitter_on.get("recall", 0.0))
        >= float(jitter_off.get("recall", 0.0)) - 0.02,
        "stage6_report_consistency": {
            entry.get("name") for entry in baseline_report.get("scenarios", [])
        }
        == {entry.get("name") for entry in current_report.get("scenarios", [])},
        "stage7_fps": curr_avg["fps"] >= 30.0,
    }

    output = {"checks": checks, "baseline_avg": base_avg, "current_avg": curr_avg}
    output_path = repo_root / "reports" / "gate_check.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
