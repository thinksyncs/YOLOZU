import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPipCLICommands(unittest.TestCase):
    def _run(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "yolozu", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )

    def test_validate_predictions_strict(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            path = root / "preds.json"
            payload = {
                "predictions": [
                    {
                        "image": "a.jpg",
                        "detections": [{"class_id": 0, "score": 1.0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
                    }
                ]
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            proc = self._run(["validate", "predictions", str(path), "--strict"], cwd=repo_root)
            if proc.returncode != 0:
                self.fail(f"validate predictions failed:\n{proc.stdout}\n{proc.stderr}")

    def test_validate_instance_seg_predictions(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            path = root / "inst.json"
            payload = [
                {
                    "image": "a.png",
                    "instances": [{"class_id": 0, "score": 1.0, "mask": "masks/a.png"}],
                }
            ]
            path.write_text(json.dumps(payload), encoding="utf-8")
            proc = self._run(["validate", "instance-seg", str(path)], cwd=repo_root)
            if proc.returncode != 0:
                self.fail(f"validate instance-seg failed:\n{proc.stdout}\n{proc.stderr}")

    def test_onnxrt_export_dry_run_writes_json(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            (dataset_root / "images" / "train2017").mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / "train2017").mkdir(parents=True, exist_ok=True)
            (dataset_root / "images" / "train2017" / "000001.jpg").write_bytes(b"")
            (dataset_root / "labels" / "train2017" / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            out_path = root / "preds_onnxrt.json"
            proc = self._run(
                [
                    "onnxrt",
                    "export",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "train2017",
                    "--max-images",
                    "1",
                    "--dry-run",
                    "--output",
                    str(out_path),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"onnxrt export --dry-run failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("predictions", payload)
            self.assertIn("meta", payload)

    def test_eval_instance_seg_smoke(self):
        repo_root = Path(__file__).resolve().parents[1]
        try:
            import numpy as _  # noqa: F401
            from PIL import Image as _  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"deps not available: {exc}")

        dataset_root = repo_root / "examples" / "instance_seg_demo" / "dataset"
        preds_path = repo_root / "examples" / "instance_seg_demo" / "predictions" / "instance_seg_predictions.json"
        pred_root = repo_root / "examples" / "instance_seg_demo" / "predictions"
        if not (dataset_root.is_dir() and preds_path.is_file()):
            self.skipTest("instance_seg_demo missing")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_json = root / "instance_seg_eval.json"
            proc = self._run(
                [
                    "eval-instance-seg",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--predictions",
                    str(preds_path),
                    "--pred-root",
                    str(pred_root),
                    "--output",
                    str(out_json),
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"eval-instance-seg failed:\n{proc.stdout}\n{proc.stderr}")
            report = json.loads(out_json.read_text(encoding="utf-8"))
            metrics = report.get("metrics") or {}
            self.assertAlmostEqual(float(metrics.get("map50")), 1.0, places=6)

    def test_demo_instance_seg_smoke(self):
        repo_root = Path(__file__).resolve().parents[1]
        try:
            import numpy as _  # noqa: F401
            from PIL import Image as _  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"deps not available: {exc}")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            run_dir = root / "run"
            proc = self._run(
                [
                    "demo",
                    "instance-seg",
                    "--num-images",
                    "3",
                    "--image-size",
                    "48",
                    "--max-instances",
                    "2",
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"demo instance-seg failed:\n{proc.stdout}\n{proc.stderr}")

            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            self.assertTrue(lines, "demo instance-seg produced no stdout")
            out_path = Path(lines[-1])
            self.assertTrue(out_path.is_file(), f"demo report missing: {out_path}")
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("kind"), "instance_seg_demo")
            res = payload.get("result") or {}
            self.assertIn("map50", res)
            self.assertIn("map50_95", res)


if __name__ == "__main__":
    unittest.main()
