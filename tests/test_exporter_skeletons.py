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
        0xD8,
        0xFF,
        0xC0,
        0x00,
        0x11,
        0x08,
        0x00,
        0x01,
        0x00,
        0x01,
        0x03,
        0x01,
        0x11,
        0x00,
        0x02,
        0x11,
        0x01,
        0x03,
        0x11,
        0x01,
        0xFF,
        0xD9,
    ]
)


class TestExporterSkeletons(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def _make_dataset(self, root: Path) -> Path:
        dataset = root / "coco-yolo"
        images = dataset / "images" / "val2017"
        labels = dataset / "labels" / "val2017"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / "000001.jpg").write_bytes(_MIN_JPEG_1X1)
        (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        return dataset

    def test_onnxrt_dry_run_writes_wrapped_json(self):
        tool = _load_module(self.repo_root / "tools" / "export_predictions_onnxrt.py", "export_predictions_onnxrt")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = self._make_dataset(tmp)
            out_path = tmp / "pred.json"
            with redirect_stdout(io.StringIO()):
                tool.main(["--dataset", str(dataset), "--dry-run", "--wrap", "--output", str(out_path)])
            doc = json.loads(out_path.read_text())
            self.assertIn("predictions", doc)
            self.assertIn("meta", doc)
            self.assertEqual(doc["meta"]["imgsz"], 640)
            self.assertTrue(doc["meta"]["dry_run"])

    def test_trt_dry_run_writes_wrapped_json(self):
        tool = _load_module(self.repo_root / "tools" / "export_predictions_trt.py", "export_predictions_trt")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = self._make_dataset(tmp)
            out_path = tmp / "pred.json"
            with redirect_stdout(io.StringIO()):
                tool.main(["--dataset", str(dataset), "--dry-run", "--wrap", "--output", str(out_path)])
            doc = json.loads(out_path.read_text())
            self.assertIn("predictions", doc)
            self.assertIn("meta", doc)
            self.assertEqual(doc["meta"]["imgsz"], 640)
            self.assertTrue(doc["meta"]["dry_run"])


if __name__ == "__main__":
    unittest.main()
