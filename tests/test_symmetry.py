import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.math3d import geodesic_distance, mat_identity, rotation_z
from yolozu.symmetry import (
    enumerate_symmetry_rotations,
    min_symmetry_geodesic,
    score_template_sym,
)


class TestSymmetry(unittest.TestCase):
    def test_symmetry_enumeration_counts(self):
        c2 = enumerate_symmetry_rotations({"type": "C2", "axis": [0.0, 0.0, 1.0]})
        c4 = enumerate_symmetry_rotations({"type": "C4", "axis": [0.0, 0.0, 1.0]})
        self.assertEqual(len(c2), 2)
        self.assertEqual(len(c4), 4)
        self.assertLess(geodesic_distance(c2[0], mat_identity()), 1e-6)

    def test_symmetry_loss_c4(self):
        r_gt = mat_identity()
        r_pred = rotation_z(1.5707963267948966)  # 90 degrees
        spec = {"type": "C4", "axis": [0.0, 0.0, 1.0]}
        loss = min_symmetry_geodesic(r_pred, r_gt, spec)
        self.assertLess(loss, 1e-6)

    def test_symmetry_loss_none(self):
        r_gt = mat_identity()
        r_pred = rotation_z(1.5707963267948966)  # 90 degrees
        spec = {"type": "none"}
        loss = min_symmetry_geodesic(r_pred, r_gt, spec)
        self.assertGreater(loss, 0.1)

    def test_symmetry_loss_cinf(self):
        r_gt = mat_identity()
        r_pred = rotation_z(0.7853981633974483)  # 45 degrees
        spec = {"type": "Cinf", "axis": [0.0, 0.0, 1.0]}
        loss = min_symmetry_geodesic(r_pred, r_gt, spec, sample_count=8)
        self.assertLess(loss, 1e-6)

    def test_template_score_max(self):
        r_gt = mat_identity()
        r_pred = rotation_z(1.5707963267948966)
        spec = {"type": "C4", "axis": [0.0, 0.0, 1.0]}

        def score_fn(r):
            return -geodesic_distance(r, r_gt)

        score = score_template_sym(score_fn, r_pred, spec)
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_template_score_symmetry_improves(self):
        r_gt = mat_identity()
        r_pred = rotation_z(1.5707963267948966)
        spec = {"type": "C4", "axis": [0.0, 0.0, 1.0]}

        def score_fn(r):
            return -geodesic_distance(r, r_gt)

        base_score = score_fn(r_pred)
        sym_score = score_template_sym(score_fn, r_pred, spec)
        self.assertGreaterEqual(sym_score, base_score)


if __name__ == "__main__":
    unittest.main()
