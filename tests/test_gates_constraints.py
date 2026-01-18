import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.constraints import (
    apply_constraints,
    depth_prior,
    depth_prior_penalty,
    is_above_plane,
    upright_violation_deg,
)
from yolozu.gates import final_score, passes_template_gate
from yolozu.math3d import rotation_matrix_axis_angle


class TestGatesConstraints(unittest.TestCase):
    def test_final_score(self):
        score = final_score(0.8, 0.5, 0.1, 0.2, {"det": 1.0, "tmp": 0.5, "unc": 1.0})
        self.assertAlmostEqual(score, 0.8 + 0.25 - 0.3, places=6)

    def test_template_gate(self):
        self.assertTrue(passes_template_gate(0.4, enabled=False, tau=0.5))
        self.assertFalse(passes_template_gate(0.4, enabled=True, tau=0.5))

    def test_depth_prior_monotonic(self):
        z_small = depth_prior((20.0, 20.0), (0.2, 0.2), (500.0, 500.0))
        z_large = depth_prior((40.0, 40.0), (0.2, 0.2), (500.0, 500.0))
        self.assertLess(z_large, z_small)
        self.assertLess(depth_prior_penalty(z_small, z_large), 1.0)

    def test_plane_and_upright(self):
        self.assertTrue(is_above_plane((0.0, 0.0, 1.0), [0.0, 0.0, 1.0], 0.0))
        r_roll = rotation_matrix_axis_angle([1.0, 0.0, 0.0], 0.7853981633974483)
        violation = upright_violation_deg(r_roll, (-10.0, 10.0), (-10.0, 10.0))
        self.assertGreater(violation, 0.0)

    def test_constraint_toggles(self):
        cfg = {
            "enabled": {"depth_prior": True, "table_plane": True, "upright": True},
            "table_plane": {"n": [0.0, 0.0, 1.0], "d": 0.0},
            "depth_prior": {"default": {"min_z": 0.5, "max_z": 1.5}, "per_class": {}},
            "upright": {
                "default": {"roll_deg": (-10.0, 10.0), "pitch_deg": (-10.0, 10.0)},
                "per_class": {},
            },
        }
        r_roll = rotation_matrix_axis_angle([1.0, 0.0, 0.0], 0.7853981633974483)
        result = apply_constraints(
            cfg,
            class_key="cup",
            bbox_wh=(20.0, 20.0),
            size_wh=(0.2, 0.2),
            intrinsics_fx_fy=(500.0, 500.0),
            t_xyz=(0.0, 0.0, -1.0),
            r_mat=r_roll,
            z_pred=2.0,
        )
        self.assertGreater(result["depth_prior_penalty"], 0.0)
        self.assertGreater(result["depth_range_violation"], 0.0)
        self.assertFalse(result["plane_ok"])
        self.assertGreater(result["upright_violation"], 0.0)

        cfg["enabled"] = {"depth_prior": False, "table_plane": False, "upright": False}
        result = apply_constraints(
            cfg,
            class_key="cup",
            bbox_wh=(20.0, 20.0),
            size_wh=(0.2, 0.2),
            intrinsics_fx_fy=(500.0, 500.0),
            t_xyz=(0.0, 0.0, -1.0),
            r_mat=r_roll,
            z_pred=2.0,
        )
        self.assertEqual(result["depth_prior_penalty"], 0.0)
        self.assertEqual(result["depth_range_violation"], 0.0)
        self.assertTrue(result["plane_ok"])
        self.assertEqual(result["upright_violation"], 0.0)


if __name__ == "__main__":
    unittest.main()
