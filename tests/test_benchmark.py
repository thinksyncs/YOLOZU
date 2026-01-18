import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.benchmark import run_benchmark


class TestBenchmark(unittest.TestCase):
    def test_benchmark_returns_fps(self):
        fps = run_benchmark(iterations=10, sleep_s=0.0)
        self.assertGreater(fps, 0.0)


if __name__ == "__main__":
    unittest.main()
