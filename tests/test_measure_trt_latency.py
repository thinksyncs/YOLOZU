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


class TestMeasureTrtLatency(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_measure_trt_latency_dry_run_writes_metrics_report(self):
        tool = _load_module(self.repo_root / "tools" / "measure_trt_latency.py", "measure_trt_latency")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "latency.json"

            with redirect_stdout(io.StringIO()):
                tool.main(
                    [
                        "--dry-run",
                        "--iterations",
                        "5",
                        "--warmup",
                        "1",
                        "--notes",
                        "unit-test",
                        "--output",
                        str(out),
                    ]
                )

            payload = json.loads(out.read_text())
            self.assertEqual(payload.get("schema_version"), 1)
            self.assertIsInstance(payload.get("metrics"), dict)
            self.assertIn("fps", payload["metrics"])
            self.assertIsInstance(payload.get("meta"), dict)
            self.assertTrue(payload["meta"].get("dry_run"))


if __name__ == "__main__":
    unittest.main()

