import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestBenchmarkEATAStabilityTool(unittest.TestCase):
    def test_generates_tradeoff_and_recommended_defaults(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "benchmark_eata_stability.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            baseline = root / "baseline.json"
            eata = root / "eata.json"

            baseline.write_text(
                json.dumps(
                    {
                        "predictions": [],
                        "meta": {
                            "ttt": {
                                "enabled": True,
                                "report": {
                                    "method": "tent",
                                    "seconds": 0.5,
                                    "losses": [1.1, 1.0],
                                    "warnings": ["max_update_norm_exceeded"],
                                    "step_metrics": [
                                        {"step": 0},
                                        {"step": 1},
                                    ],
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            eata.write_text(
                json.dumps(
                    {
                        "predictions": [],
                        "meta": {
                            "ttt": {
                                "enabled": True,
                                "report": {
                                    "method": "eata",
                                    "seconds": 0.65,
                                    "losses": [0.95, 0.9],
                                    "warnings": [],
                                    "step_metrics": [
                                        {"step": 0, "selected_ratio": 0.5},
                                        {"step": 1, "selected_ratio": 0.6},
                                    ],
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            out_json = root / "bench.json"
            out_md = root / "bench.md"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--baseline",
                    str(baseline.relative_to(repo_root)),
                    "--eata",
                    str(eata.relative_to(repo_root)),
                    "--output-json",
                    str(out_json.relative_to(repo_root)),
                    "--output-md",
                    str(out_md.relative_to(repo_root)),
                    "--max-overhead-ratio",
                    "1.5",
                    "--max-loss-ratio",
                    "1.05",
                    "--min-selected-ratio",
                    "0.1",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"benchmark_eata_stability.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("kind"), "eata_stability_efficiency_benchmark")
            self.assertIn("tradeoff", payload)
            self.assertIn("recommended_defaults", payload)
            self.assertTrue(payload.get("recommended_defaults", {}).get("enabled"))

            md = out_md.read_text(encoding="utf-8")
            self.assertIn("EATA stability/efficiency benchmark", md)
            self.assertIn("Recommended defaults", md)


if __name__ == "__main__":
    unittest.main()
