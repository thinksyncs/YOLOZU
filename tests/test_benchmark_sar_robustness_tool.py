from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _wrapped_report(*, losses: list[float], seconds: float, warnings: list[str]) -> dict:
    return {
        "meta": {
            "ttt": {
                "report": {
                    "mode": "stream",
                    "losses": losses,
                    "seconds": seconds,
                    "warnings": warnings,
                }
            }
        }
    }


def test_benchmark_sar_robustness_go_decision(tmp_path: Path) -> None:
    cotta = tmp_path / "cotta.json"
    eata = tmp_path / "eata.json"
    sar = tmp_path / "sar.json"

    cotta.write_text(
        json.dumps(_wrapped_report(losses=[1.00, 0.95, 0.90], seconds=0.12, warnings=[])),
        encoding="utf-8",
    )
    eata.write_text(
        json.dumps(_wrapped_report(losses=[0.98, 0.92, 0.89], seconds=0.10, warnings=[])),
        encoding="utf-8",
    )
    sar.write_text(
        json.dumps(_wrapped_report(losses=[0.96, 0.90, 0.87], seconds=0.11, warnings=[])),
        encoding="utf-8",
    )

    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    subprocess.run(
        [
            sys.executable,
            "tools/benchmark_sar_robustness.py",
            "--cotta",
            str(cotta),
            "--eata",
            str(eata),
            "--sar",
            str(sar),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--max-overhead-ratio",
            "1.3",
            "--max-loss-ratio",
            "1.05",
            "--max-variance-ratio",
            "1.2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["kind"] == "sar_robustness_impact"
    assert report["checks"]["robustness_gain"] is True
    assert report["decision"]["go"] is True
    assert report["decision"]["summary"] == "go"

    markdown = out_md.read_text(encoding="utf-8")
    assert "# SAR robustness impact report" in markdown
    assert "| sar |" in markdown


def test_benchmark_sar_robustness_no_go_on_overhead_and_warnings(tmp_path: Path) -> None:
    cotta = tmp_path / "cotta.json"
    eata = tmp_path / "eata.json"
    sar = tmp_path / "sar.json"

    cotta.write_text(
        json.dumps(_wrapped_report(losses=[1.00, 0.95, 0.91], seconds=0.10, warnings=[])),
        encoding="utf-8",
    )
    eata.write_text(
        json.dumps(_wrapped_report(losses=[0.99, 0.96, 0.92], seconds=0.09, warnings=[])),
        encoding="utf-8",
    )
    sar.write_text(
        json.dumps(
            _wrapped_report(
                losses=[1.05, 1.01, 0.98],
                seconds=0.25,
                warnings=["grad_norm_exceeded: 1.5 > 1.0"],
            )
        ),
        encoding="utf-8",
    )

    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    subprocess.run(
        [
            sys.executable,
            "tools/benchmark_sar_robustness.py",
            "--cotta",
            str(cotta),
            "--eata",
            str(eata),
            "--sar",
            str(sar),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--max-overhead-ratio",
            "1.2",
            "--max-loss-ratio",
            "1.01",
            "--max-variance-ratio",
            "1.1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["checks"]["robustness_gain"] is False
    assert report["checks"]["acceptable_overhead"] is False
    assert report["decision"]["go"] is False
    assert report["decision"]["summary"] == "no-go"
