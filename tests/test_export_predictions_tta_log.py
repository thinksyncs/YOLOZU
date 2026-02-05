import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import export_predictions


class TestExportPredictionsTTALog(unittest.TestCase):
    def test_summarize_tta(self):
        preds = [
            {"image": "x.jpg", "detections": [], "tta_mask": [True, False, True]},
            {"image": "y.jpg", "detections": [], "tta_mask": [False]},
        ]
        summary = export_predictions._summarize_tta(preds, warnings=["w1"])
        self.assertEqual(summary["detections"], 4)
        self.assertEqual(summary["applied"], 2)
        self.assertAlmostEqual(summary["applied_ratio"], 0.5)
        self.assertEqual(summary["warnings"], ["w1"])


if __name__ == "__main__":
    unittest.main()
