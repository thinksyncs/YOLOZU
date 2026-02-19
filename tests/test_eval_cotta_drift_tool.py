import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestEvalCoTTADriftTool(unittest.TestCase):
    def test_generates_stabilization_report(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "eval_cotta_drift.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            baseline = root / "baseline.json"
            cotta = root / "cotta.json"

            baseline_payload = {
                "predictions": [],
                "meta": {
                    "ttt": {
                        "enabled": True,
                        "report": {
                            "method": "tent",
                            "losses": [1.2, 1.1],
                            "warnings": ["max_update_norm_exceeded"],
                            "stopped_early": True,
                            "step_metrics": [
                                {"total_update_norm": 0.8},
                                {"total_update_norm": 1.1},
                            ],
                        },
                    }
                },
            }
            cotta_payload = {
                "predictions": [],
                "meta": {
                    "ttt": {
                        "enabled": True,
                        "report": {
                            "method": "cotta",
                            "losses": [1.0, 0.9],
                            "warnings": [],
                            "stopped_early": False,
                            "step_metrics": [
                                {"total_update_norm": 0.5},
                                {"total_update_norm": 0.7},
                            ],
                        },
                    }
                },
            }
            baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")
            cotta.write_text(json.dumps(cotta_payload), encoding="utf-8")

            out_json = root / "report.json"
            out_md = root / "report.md"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--baseline",
                    str(baseline.relative_to(repo_root)),
                    "--cotta",
                    str(cotta.relative_to(repo_root)),
                    "--output-json",
                    str(out_json.relative_to(repo_root)),
                    "--output-md",
                    str(out_md.relative_to(repo_root)),
                    "--max-safe-total-update-norm",
                    "2.0",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"eval_cotta_drift.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("kind"), "cotta_drift_validation")
            self.assertTrue(payload.get("decision", {}).get("stabilization_pass"))
            self.assertFalse(payload.get("decision", {}).get("unsafe_drift_detected"))

            md = out_md.read_text(encoding="utf-8")
            self.assertIn("CoTTA drift validation report", md)
            self.assertIn("baseline", md)
            self.assertIn("cotta", md)

    def test_flags_unsafe_drift(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "eval_cotta_drift.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            baseline = root / "baseline.json"
            cotta = root / "cotta.json"

            baseline.write_text(
                json.dumps(
                    {
                        "predictions": [],
                        "meta": {
                            "ttt": {
                                "enabled": True,
                                "report": {
                                    "method": "tent",
                                    "losses": [1.0],
                                    "warnings": [],
                                    "step_metrics": [{"total_update_norm": 0.5}],
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            cotta.write_text(
                json.dumps(
                    {
                        "predictions": [],
                        "meta": {
                            "ttt": {
                                "enabled": True,
                                "report": {
                                    "method": "cotta",
                                    "losses": [0.9],
                                    "warnings": [],
                                    "step_metrics": [{"total_update_norm": 4.0}],
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            out_json = root / "report.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--baseline",
                    str(baseline.relative_to(repo_root)),
                    "--cotta",
                    str(cotta.relative_to(repo_root)),
                    "--output-json",
                    str(out_json.relative_to(repo_root)),
                    "--max-safe-total-update-norm",
                    "1.0",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"eval_cotta_drift.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(payload.get("decision", {}).get("unsafe_drift_detected"))
            self.assertFalse(payload.get("decision", {}).get("stabilization_pass"))


if __name__ == "__main__":
    unittest.main()
