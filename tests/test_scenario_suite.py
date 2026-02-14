import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.scenario_suite import build_report, SCENARIOS


class TestScenarioSuite(unittest.TestCase):
    def test_report_schema(self):
        report = build_report()
        self.assertIn("schema_version", report)
        self.assertIn("timestamp", report)
        self.assertIn("summary", report)
        self.assertIn("scenarios", report)
        self.assertEqual(len(report["scenarios"]), len(SCENARIOS))
        for entry in report["scenarios"]:
            self.assertIn("name", entry)
            metrics = entry.get("metrics", {})
            for key in ("fps", "map", "recall", "depth_error", "pose_error", "rejection_rate"):
                self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
