import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

repo_root = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "torch not installed")
class TestRefinePredictionsHessianCLI(unittest.TestCase):
    def test_cli_smoke_refine_offsets(self):
        script = repo_root / "tools" / "refine_predictions_hessian.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            preds_path = root / "preds.json"
            out_path = root / "preds_refined.json"

            preds = [
                {
                    "image": "000001.jpg",
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "log_z": 0.0,
                            "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            "offsets": [10.0, 10.0],
                        }
                    ],
                }
            ]
            preds_path.write_text(json.dumps({"predictions": preds}, indent=2))

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--predictions",
                    str(preds_path),
                    "--output",
                    str(out_path),
                    "--refine-offsets",
                    "--wrap",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"refine_predictions_hessian.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text())
            self.assertIn("predictions", payload)
            det0 = payload["predictions"][0]["detections"][0]
            self.assertIn("hessian_refinement", det0)


if __name__ == "__main__":
    unittest.main()

