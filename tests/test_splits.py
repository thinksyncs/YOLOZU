import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.splits import deterministic_split_paths


class TestSplits(unittest.TestCase):
    def test_deterministic_split_is_stable(self):
        paths = [f"img_{i:03d}.jpg" for i in range(100)]
        a = deterministic_split_paths(paths, val_fraction=0.2, seed=123)
        b = deterministic_split_paths(paths, val_fraction=0.2, seed=123)
        self.assertEqual(a, b)

    def test_seed_changes_split(self):
        paths = [f"img_{i:03d}.jpg" for i in range(200)]
        a = deterministic_split_paths(paths, val_fraction=0.2, seed=1)
        b = deterministic_split_paths(paths, val_fraction=0.2, seed=2)
        self.assertNotEqual(a.val, b.val)


if __name__ == "__main__":
    unittest.main()

