import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.adapter import DummyAdapter
from yolozu.runner import run_adapter


class TestRunner(unittest.TestCase):
    def test_run_adapter_keys(self):
        adapter = DummyAdapter()
        records = [{"image": "fake.jpg", "labels": []}]
        result = run_adapter(adapter, records)
        self.assertIn("images", result)
        self.assertIn("detections", result)
        self.assertEqual(result["images"], 1)


if __name__ == "__main__":
    unittest.main()
