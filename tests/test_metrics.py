import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.math3d import mat_identity, rotation_z
from yolozu.metrics import add_s, symmetry_geodesic


class TestMetrics(unittest.TestCase):
    def test_symmetry_geodesic(self):
        r_gt = mat_identity()
        r_pred = rotation_z(1.5707963267948966)
        loss = symmetry_geodesic(r_pred, r_gt, {"type": "C4", "axis": [0.0, 0.0, 1.0]})
        self.assertLess(loss, 1e-6)

    def test_add_s_symmetry(self):
        points = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        r_gt = mat_identity()
        t_gt = (0.0, 0.0, 0.0)
        r_pred = rotation_z(1.5707963267948966)
        t_pred = (0.0, 0.0, 0.0)

        err_sym = add_s(
            points,
            r_pred,
            t_pred,
            r_gt,
            t_gt,
            {"type": "C4", "axis": [0.0, 0.0, 1.0]},
        )
        err_none = add_s(points, r_pred, t_pred, r_gt, t_gt, {"type": "none"})
        self.assertLess(err_sym, 1e-6)
        self.assertGreater(err_none, 0.1)


if __name__ == "__main__":
    unittest.main()
