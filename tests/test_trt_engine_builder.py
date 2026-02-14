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


class TestTrtEngineBuilder(unittest.TestCase):
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

    def test_build_trt_engine_dry_run_writes_meta_and_calib_list(self):
        tool = _load_module(self.repo_root / "tools" / "build_trt_engine.py", "build_trt_engine")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = self._make_dataset(tmp)
            onnx_path = tmp / "model.onnx"
            onnx_path.write_bytes(b"dummy")
            meta_path = tmp / "meta.json"
            calib_list = tmp / "calib.txt"

            with redirect_stdout(io.StringIO()):
                tool.main(
                    [
                        "--onnx",
                        str(onnx_path),
                        "--engine",
                        str(tmp / "model.plan"),
                        "--precision",
                        "int8",
                        "--calib-cache",
                        str(tmp / "calib.cache"),
                        "--calib-dataset",
                        str(dataset),
                        "--calib-images",
                        "1",
                        "--calib-list-output",
                        str(calib_list),
                        "--meta-output",
                        str(meta_path),
                        "--dry-run",
                    ]
                )

            meta = json.loads(meta_path.read_text())
            self.assertEqual(meta["precision"], "int8")
            self.assertTrue(Path(meta["onnx"]).exists())
            self.assertTrue(calib_list.exists())
            self.assertIn("--int8", meta["command_str"])


if __name__ == "__main__":
    unittest.main()