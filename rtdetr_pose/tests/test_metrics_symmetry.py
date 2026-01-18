import math
import sys
from pathlib import Path
import unittest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rtdetr_pose.metrics import symmetry_geodesic


class TestMetricsSymmetry(unittest.TestCase):
    def test_symmetry_geodesic(self):
        r_gt = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        rot_180_z = [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        sym_rots = [r_gt, rot_180_z]
        value = symmetry_geodesic(rot_180_z, r_gt, sym_rots)
        self.assertAlmostEqual(value, 0.0, places=6)

        value_non_sym = symmetry_geodesic(rot_180_z, r_gt, [r_gt])
        self.assertAlmostEqual(value_non_sym, math.pi, places=6)


if __name__ == "__main__":
    unittest.main()
