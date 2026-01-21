import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.adapter import PrecomputedAdapter


class TestPrecomputedAdapter(unittest.TestCase):
    def test_matches_by_full_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "preds.json"
            payload = [
                {"image": "/tmp/a.jpg", "detections": [{"score": 0.9}]},
            ]
            path.write_text(json.dumps(payload))
            adapter = PrecomputedAdapter(path)
            out = adapter.predict([{"image": "/tmp/a.jpg", "labels": []}])
            self.assertEqual(out[0]["detections"], [{"score": 0.9}])

    def test_matches_by_basename(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "preds.json"
            payload = {"0001.jpg": [{"score": 0.5}]}
            path.write_text(json.dumps(payload))
            adapter = PrecomputedAdapter(path)
            out = adapter.predict([{"image": "/any/dir/0001.jpg", "labels": []}])
            self.assertEqual(out[0]["detections"], [{"score": 0.5}])

    def test_top_level_predictions_key(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "preds.json"
            payload = {"predictions": [{"image": "x.jpg", "detections": []}]}
            path.write_text(json.dumps(payload))
            adapter = PrecomputedAdapter(path)
            out = adapter.predict([{"image": "x.jpg", "labels": []}])
            self.assertEqual(out[0]["detections"], [])


if __name__ == "__main__":
    unittest.main()
