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


class TestRunTrtPipeline(unittest.TestCase):
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

    def test_run_trt_pipeline_dry_run_produces_artifacts(self):
        tool = _load_module(self.repo_root / "tools" / "run_trt_pipeline.py", "run_trt_pipeline")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = self._make_dataset(tmp)

            models = tmp / "models"
            models.mkdir(parents=True, exist_ok=True)
            (models / "yolo26n.onnx").write_bytes(b"dummy")

            engines = tmp / "engines"
            reports = tmp / "reports"
            runs = tmp / "runs"
            engines.mkdir(parents=True, exist_ok=True)
            reports.mkdir(parents=True, exist_ok=True)
            runs.mkdir(parents=True, exist_ok=True)

            run_id = "unit-test"
            run_dir = runs / "{run_id}"

            with redirect_stdout(io.StringIO()):
                tool.main(
                    [
                        "--dry-run",
                        "--dataset",
                        str(dataset),
                        "--buckets",
                        "yolo26n",
                        "--onnx-template",
                        str(models / "{bucket}.onnx"),
                        "--engine-template",
                        str(engines / "{bucket}_{precision}.plan"),
                        "--engine-meta-template",
                        str(reports / "trt_engine_{bucket}_{precision}.json"),
                        "--pred-onnxrt-template",
                        str(reports / "pred_onnxrt_{bucket}.json"),
                        "--pred-trt-template",
                        str(reports / "pred_trt_{bucket}.json"),
                        "--parity-report-template",
                        str(reports / "parity_{bucket}.json"),
                        "--eval-suite-output",
                        str(reports / "eval_suite_trt.json"),
                        "--latency-template",
                        str(reports / "latency_{bucket}.json"),
                        "--benchmark-config",
                        str(reports / "benchmark_cfg.json"),
                        "--benchmark-output",
                        str(reports / "benchmark_latency.json"),
                        "--benchmark-history",
                        str(reports / "benchmark_latency.jsonl"),
                        "--timing-cache",
                        str(engines / "timing.cache"),
                        "--run-id",
                        run_id,
                        "--run-dir",
                        str(run_dir),
                    ]
                )

            # Core artifacts
            self.assertTrue((reports / "trt_engine_yolo26n_fp16.json").exists())
            self.assertTrue((reports / "pred_onnxrt_yolo26n.json").exists())
            self.assertTrue((reports / "pred_trt_yolo26n.json").exists())
            self.assertTrue((reports / "parity_yolo26n.json").exists())
            self.assertTrue((reports / "eval_suite_trt.json").exists())
            self.assertTrue((reports / "latency_yolo26n.json").exists())
            self.assertTrue((reports / "benchmark_latency.json").exists())

            # Run record
            final_run_dir = runs / run_id
            self.assertTrue((final_run_dir / "run.json").exists())
            run_payload = json.loads((final_run_dir / "run.json").read_text())
            self.assertEqual(run_payload.get("run_id"), run_id)
            self.assertEqual(run_payload.get("buckets"), ["yolo26n"])
            self.assertTrue(run_payload.get("dry_run"))


if __name__ == "__main__":
    unittest.main()

