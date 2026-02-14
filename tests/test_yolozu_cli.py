import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestYOLOZUCLI(unittest.TestCase):
    def test_help_lists_continual_commands(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"yolozu --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("continual-train", proc.stdout)
        self.assertIn("continual-eval", proc.stdout)

    def test_doctor_writes_json(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_path = root / "doctor.json"
            proc = subprocess.run(
                ["python3", str(script), "doctor", "--output", str(out_path.relative_to(repo_root))],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu doctor failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text())
            self.assertIn("timestamp", payload)
            self.assertIn("gpu", payload)
            self.assertIn("env", payload)

    def test_export_dummy_injects_run_meta(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            # Dummy adapter doesn't open images; an empty file is fine.
            (images / "000001.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            out_path = root / "preds.json"
            proc = subprocess.run(
                [
                    "python3",
                    str(script),
                    "export",
                    "--backend",
                    "dummy",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "train2017",
                    "--max-images",
                    "1",
                    "--output",
                    str(out_path),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu export failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text())
            self.assertIn("predictions", payload)
            meta = payload.get("meta") or {}
            self.assertIn("run", meta)
            run = meta.get("run") or {}
            self.assertIn("config_hash", run)
            self.assertIn("git", run)
            self.assertIn("env", run)

    def test_export_cache_reuses_by_fingerprint(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / "000001.jpg").write_bytes(b"")

            cache_dir = root / "cache"

            cmd = [
                "python3",
                str(script),
                "export",
                "--backend",
                "dummy",
                "--dataset",
                str(dataset_root),
                "--split",
                "train2017",
                "--max-images",
                "1",
                "--cache",
                "--cache-dir",
                str(cache_dir),
            ]
            proc1 = subprocess.run(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc1.returncode != 0:
                self.fail(f"yolozu export --cache failed:\n{proc1.stdout}\n{proc1.stderr}")
            out_path = Path(proc1.stdout.strip().splitlines()[-1])
            self.assertTrue(out_path.is_file())

            payload1 = json.loads(out_path.read_text())
            ts1 = payload1.get("meta", {}).get("run", {}).get("timestamp")
            self.assertIsInstance(ts1, str)

            run_cfg = out_path.parent / "run_config.json"
            self.assertTrue(run_cfg.is_file())
            cfg_payload = json.loads(run_cfg.read_text())
            self.assertEqual(cfg_payload.get("config_hash"), payload1.get("meta", {}).get("run", {}).get("config_hash"))

            proc2 = subprocess.run(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc2.returncode != 0:
                self.fail(f"yolozu export --cache (2nd run) failed:\n{proc2.stdout}\n{proc2.stderr}")
            out_path2 = Path(proc2.stdout.strip().splitlines()[-1])
            self.assertEqual(out_path2, out_path)

            payload2 = json.loads(out_path2.read_text())
            ts2 = payload2.get("meta", {}).get("run", {}).get("timestamp")
            self.assertEqual(ts2, ts1)

    def test_sweep_wrapper_dry_run(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        proc = subprocess.run(
            ["python3", str(script), "sweep", "--config", "docs/hpo_sweep_example.json", "--dry-run", "--max-runs", "1"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"yolozu sweep failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("python3 tools/mock_train.py", proc.stdout)

    def test_predict_images_dummy_writes_overlays_and_html(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"PIL not available: {exc}")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            input_dir = root / "images"
            input_dir.mkdir(parents=True, exist_ok=True)
            img_path = input_dir / "a.png"
            Image.new("RGB", (16, 16), color=(0, 0, 0)).save(img_path)

            out_path = root / "preds.json"
            overlays_dir = root / "overlays"
            html_path = root / "report.html"

            proc = subprocess.run(
                [
                    "python3",
                    str(script),
                    "predict-images",
                    "--backend",
                    "dummy",
                    "--input-dir",
                    str(input_dir),
                    "--max-images",
                    "1",
                    "--output",
                    str(out_path),
                    "--overlays-dir",
                    str(overlays_dir),
                    "--html",
                    str(html_path),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu predict-images failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text())
            preds = payload.get("predictions") or []
            self.assertEqual(len(preds), 1)
            self.assertEqual(Path(preds[0]["image"]), img_path)

            self.assertTrue(html_path.is_file())
            self.assertTrue(overlays_dir.is_dir())
            overlays = list(overlays_dir.glob("*.png"))
            self.assertTrue(overlays, "expected at least one overlay image")

    def test_eval_instance_seg_demo_writes_html_and_overlays(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        try:
            import numpy as _  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"deps not available: {exc}")

        dataset_root = repo_root / "examples" / "instance_seg_demo" / "dataset"
        preds_path = repo_root / "examples" / "instance_seg_demo" / "predictions" / "instance_seg_predictions.json"
        pred_root = repo_root / "examples" / "instance_seg_demo" / "predictions"
        classes_path = repo_root / "examples" / "instance_seg_demo" / "classes.txt"
        if not (dataset_root.is_dir() and preds_path.is_file()):
            self.skipTest("instance_seg_demo missing")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_json = root / "instance_seg_eval.json"
            out_html = root / "instance_seg_eval.html"
            overlays_dir = root / "overlays"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "eval-instance-seg",
                    "--dataset",
                    str(dataset_root.relative_to(repo_root)),
                    "--split",
                    "val2017",
                    "--predictions",
                    str(preds_path.relative_to(repo_root)),
                    "--pred-root",
                    str(pred_root.relative_to(repo_root)),
                    "--classes",
                    str(classes_path.relative_to(repo_root)),
                    "--output",
                    str(out_json.relative_to(repo_root)),
                    "--html",
                    str(out_html.relative_to(repo_root)),
                    "--overlays-dir",
                    str(overlays_dir.relative_to(repo_root)),
                    "--max-overlays",
                    "1",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu eval-instance-seg failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_json.is_file())
            report = json.loads(out_json.read_text(encoding="utf-8"))
            metrics = report.get("metrics") or {}
            self.assertAlmostEqual(float(metrics.get("map50")), 1.0, places=6)

            self.assertTrue(out_html.is_file())
            self.assertTrue(overlays_dir.is_dir())
            overlays = list(overlays_dir.glob("*.png"))
            self.assertTrue(overlays, "expected at least one overlay image")


if __name__ == "__main__":
    unittest.main()
