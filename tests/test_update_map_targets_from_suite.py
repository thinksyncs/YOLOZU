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


class TestUpdateMapTargetsFromSuite(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_updates_targets_from_suite(self):
        tool = _load_module(self.repo_root / "tools" / "update_map_targets_from_suite.py", "update_map_targets_from_suite")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            suite = {
                "timestamp": "2026-02-14T00:00:00Z",
                "protocol_id": "yolo26",
                "results": [
                    {"name": "pred_trt_yolo26n", "path": "reports/pred_trt_yolo26n.json", "metrics": {"map50_95": 0.123}},
                    {"name": "pred_trt_yolo26s", "path": "reports/pred_trt_yolo26s.json", "metrics": {"map50_95": 0.234}},
                ],
            }
            suite_path = tmp / "eval_suite.json"
            suite_path.write_text(json.dumps(suite))

            targets = {
                "protocol_id": "yolo26",
                "metric_key": "map50_95",
                "imgsz": 640,
                "targets": {"yolo26n": None, "yolo26s": None, "yolo26m": None, "yolo26l": None, "yolo26x": None},
            }
            targets_path = tmp / "yolo26_targets.json"
            targets_path.write_text(json.dumps(targets))

            out = io.StringIO()
            with redirect_stdout(out):
                tool.main(["--suite", str(suite_path), "--targets", str(targets_path)])

            updated = json.loads(targets_path.read_text())
            self.assertAlmostEqual(updated["targets"]["yolo26n"], 0.123, places=6)
            self.assertAlmostEqual(updated["targets"]["yolo26s"], 0.234, places=6)
            self.assertIn("provenance", updated)
            self.assertEqual(updated["provenance"]["metric_key"], "map50_95")


if __name__ == "__main__":
    unittest.main()

