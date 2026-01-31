import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MIN_JPEG_1X1 = bytes(
    [
        0xFF,
        0xD8,  # SOI
        0xFF,
        0xC0,  # SOF0
        0x00,
        0x11,  # length
        0x08,  # precision
        0x00,
        0x01,  # height = 1
        0x00,
        0x01,  # width = 1
        0x03,  # components
        0x01,
        0x11,
        0x00,  # comp 1
        0x02,
        0x11,
        0x01,  # comp 2
        0x03,
        0x11,
        0x01,  # comp 3
        0xFF,
        0xD9,  # EOI
    ]
)


class TestYOLO26BaselineImport(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_import_script_archives_suite(self):
        import_tool = _load_module(self.repo_root / "tools" / "import_yolo26_baseline.py", "import_yolo26_baseline")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "coco-yolo"
            images = dataset / "images" / "val2017"
            labels = dataset / "labels" / "val2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            img1 = images / "000001.jpg"
            img2 = images / "000002.jpg"
            img1.write_bytes(_MIN_JPEG_1X1)
            img2.write_bytes(_MIN_JPEG_1X1)
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (labels / "000002.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            def write_preds(path: Path):
                preds = [
                    {
                        "image": str(img1),
                        "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
                    },
                    {
                        "image": str(img2),
                        "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
                    },
                ]
                path.write_text(json.dumps(preds))

            pred_dir = tmp / "preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            for bucket in ("n", "s", "m", "l", "x"):
                write_preds(pred_dir / f"pred_yolo26{bucket}.json")

            suite_out = tmp / "eval_suite.json"
            archive_root = tmp / "archives"

            with redirect_stdout(io.StringIO()):
                import_tool.main(
                    [
                        "--dataset",
                        str(dataset),
                        "--predictions-glob",
                        str(pred_dir / "pred_yolo26*.json"),
                        "--output-suite",
                        str(suite_out),
                        "--archive-root",
                        str(archive_root),
                        "--run-id",
                        "test-run",
                        "--dry-run",
                        "--notes",
                        "unit test",
                    ]
                )

            run_dir = archive_root / "test-run"
            self.assertTrue((run_dir / "eval_suite.json").exists())
            self.assertTrue((run_dir / "run.json").exists())

            suite = json.loads((run_dir / "eval_suite.json").read_text())
            self.assertEqual(suite.get("protocol_id"), "yolo26")
            self.assertEqual(len(suite.get("results", [])), 5)

            run = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(run.get("protocol_id"), "yolo26")
            self.assertEqual(run.get("run_id"), "test-run")
            self.assertEqual(len(run.get("predictions", [])), 5)
            self.assertIn("bucket_files", run)
            self.assertIn("yolo26n", run["bucket_files"])


if __name__ == "__main__":
    unittest.main()
