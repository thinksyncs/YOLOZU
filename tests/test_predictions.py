import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.predictions import (
    load_predictions_entries,
    load_predictions_index,
    validate_predictions_entries,
)


class TestPredictionsIO(unittest.TestCase):
    def test_load_entries_list_format(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preds.json"
            payload = [{"image": "a.jpg", "detections": [{"score": 0.1, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}]}]
            p.write_text(json.dumps(payload))
            entries = load_predictions_entries(p)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["image"], "a.jpg")

    def test_load_entries_wrapped_format(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preds.json"
            payload = {"predictions": [{"image": "a.jpg", "detections": []}], "meta": {"x": 1}}
            p.write_text(json.dumps(payload))
            entries = load_predictions_entries(p)
            self.assertEqual(entries[0]["image"], "a.jpg")

    def test_load_index_adds_basename_alias(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preds.json"
            payload = [{"image": "/tmp/0001.jpg", "detections": [{"score": 0.9, "bbox": {"cx": 0.1, "cy": 0.1, "w": 0.1, "h": 0.1}}]}]
            p.write_text(json.dumps(payload))
            idx = load_predictions_index(p)
            self.assertIn("/tmp/0001.jpg", idx)
            self.assertIn("0001.jpg", idx)

    def test_validator_warnings_non_strict(self):
        entries = [{"image": "a.jpg", "detections": [{"score": 0.1, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}]}]
        res = validate_predictions_entries(entries, strict=False)
        self.assertTrue(any("missing 'class_id'" in w for w in res.warnings))

    def test_validator_strict_requires_numeric(self):
        entries = [{"image": "a.jpg", "detections": [{"class_id": 1, "score": "0.1", "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}]}]
        with self.assertRaises(ValueError):
            validate_predictions_entries(entries, strict=True)


if __name__ == "__main__":
    unittest.main()
