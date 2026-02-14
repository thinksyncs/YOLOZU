import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.benchmark import measure_latency


class TestBenchmarkLatency(unittest.TestCase):
    def test_measure_latency_schema(self):
        metrics = measure_latency(iterations=5, warmup=1, sleep_s=0.0)
        self.assertIn("fps", metrics)
        self.assertIn("latency_ms", metrics)
        latency = metrics["latency_ms"]
        for key in ("mean", "p50", "p90", "p95", "p99", "min", "max"):
            self.assertIn(key, latency)


if __name__ == "__main__":
    unittest.main()
