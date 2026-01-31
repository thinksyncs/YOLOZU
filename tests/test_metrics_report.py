import csv
import tempfile
import unittest
from pathlib import Path

from yolozu.metrics_report import build_report, flatten_metrics, write_csv_row


class TestMetricsReport(unittest.TestCase):
    def test_flatten_metrics(self):
        flat = flatten_metrics({"losses": {"loss": 1.0, "loss_box": 2.0}, "metrics": {"map50_95": 0.5}})
        self.assertEqual(flat["losses.loss"], 1.0)
        self.assertEqual(flat["losses.loss_box"], 2.0)
        self.assertEqual(flat["metrics.map50_95"], 0.5)

    def test_write_csv_row(self):
        report = build_report(losses={"loss": 1.0}, metrics={"map50_95": 0.5}, meta={"kind": "unit"})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "m.csv"
            write_csv_row(path, report)
            with path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertIn("schema_version", row)
            self.assertIn("metrics.map50_95", row)
            self.assertIn("losses.loss", row)


if __name__ == "__main__":
    unittest.main()
