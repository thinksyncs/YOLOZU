import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestBenchmarkKeypointsEvalTool(unittest.TestCase):
    def test_benchmark_keypoints_eval_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "benchmark_keypoints_eval.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            (root / "images" / "train2017").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "train2017").mkdir(parents=True, exist_ok=True)

            img_path = root / "images" / "train2017" / "000000000001.jpg"
            img_path.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG marker bytes

            (root / "labels" / "train2017" / "000000000001.txt").write_text(
                "0 0.5 0.5 0.4 0.4 0.50 0.50 2 0.60 0.50 2\n",
                encoding="utf-8",
            )

            pred_path = root / "predictions.json"
            pred_path.write_text(
                json.dumps(
                    [
                        {
                            "image": str(img_path),
                            "detections": [
                                {
                                    "class_id": 0,
                                    "score": 1.0,
                                    "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4},
                                    "keypoints": [0.52, 0.50, 2, 0.60, 0.50, 2],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            out_path = root / "bench.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(root),
                    "--split",
                    "train2017",
                    "--predictions",
                    str(pred_path),
                    "--output",
                    str(out_path),
                    "--warmup",
                    "0",
                    "--iterations",
                    "2",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"benchmark_keypoints_eval.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics") or {}
            self.assertIn("pck", metrics)
            self.assertIn("benchmark", metrics)


if __name__ == "__main__":
    unittest.main()
