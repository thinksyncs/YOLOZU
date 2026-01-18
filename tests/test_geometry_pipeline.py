import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.geometry import corrected_intrinsics, recover_translation
from yolozu.jitter import jitter_off
from yolozu.math3d import rotation_matrix_axis_angle
from yolozu.pipeline import evaluate_candidate


class TestGeometryPipeline(unittest.TestCase):
    def test_corrected_intrinsics(self):
        k = (500.0, 500.0, 320.0, 240.0)
        delta = (0.0, 0.0, 0.0, 0.0)
        self.assertEqual(corrected_intrinsics(k, delta), k)
        delta = (0.1, -0.1, 5.0, -3.0)
        k_prime = corrected_intrinsics(k, delta)
        self.assertAlmostEqual(k_prime[0], 550.0, places=6)
        self.assertAlmostEqual(k_prime[1], 450.0, places=6)
        self.assertAlmostEqual(k_prime[2], 325.0, places=6)
        self.assertAlmostEqual(k_prime[3], 237.0, places=6)

    def test_recover_translation_uses_offsets(self):
        k = (500.0, 500.0, 320.0, 240.0)
        t0 = recover_translation((320.0, 240.0), (0.0, 0.0), 2.0, k)
        t1 = recover_translation((320.0, 240.0), (10.0, -10.0), 2.0, k)
        self.assertNotEqual(t0, t1)

    def test_evaluate_candidate_uses_k_prime(self):
        cfg = {"enabled": {"depth_prior": True, "table_plane": False, "upright": False}}
        r_mat = rotation_matrix_axis_angle([0.0, 0.0, 1.0], 0.0)
        base = evaluate_candidate(
            cfg,
            class_key="sample",
            bbox_center=(320.0, 240.0),
            bbox_wh=(20.0, 20.0),
            offsets=(0.0, 0.0),
            z_pred=2.0,
            size_wh=(0.2, 0.2),
            k=(500.0, 500.0, 320.0, 240.0),
            k_delta=(0.0, 0.0, 0.0, 0.0),
            r_mat=r_mat,
        )
        jittered = evaluate_candidate(
            cfg,
            class_key="sample",
            bbox_center=(320.0, 240.0),
            bbox_wh=(20.0, 20.0),
            offsets=(0.0, 0.0),
            z_pred=2.0,
            size_wh=(0.2, 0.2),
            k=(500.0, 500.0, 320.0, 240.0),
            k_delta=(0.1, 0.0, 0.0, 0.0),
            r_mat=r_mat,
        )
        self.assertNotEqual(base["k_prime"], jittered["k_prime"])
        self.assertNotEqual(base["constraints"]["depth_prior_penalty"], jittered["constraints"]["depth_prior_penalty"])

    def test_jitter_off_matches_baseline(self):
        cfg = {"enabled": {"depth_prior": True, "table_plane": False, "upright": False}}
        r_mat = rotation_matrix_axis_angle([0.0, 0.0, 1.0], 0.0)
        k = (500.0, 500.0, 320.0, 240.0)
        delta = jitter_off()
        baseline = evaluate_candidate(
            cfg,
            class_key="sample",
            bbox_center=(320.0, 240.0),
            bbox_wh=(20.0, 20.0),
            offsets=(0.0, 0.0),
            z_pred=2.0,
            size_wh=(0.2, 0.2),
            k=k,
            k_delta=(delta["dfx"], delta["dfy"], delta["dcx"], delta["dcy"]),
            r_mat=r_mat,
        )
        direct = evaluate_candidate(
            cfg,
            class_key="sample",
            bbox_center=(320.0, 240.0),
            bbox_wh=(20.0, 20.0),
            offsets=(0.0, 0.0),
            z_pred=2.0,
            size_wh=(0.2, 0.2),
            k=k,
            k_delta=(0.0, 0.0, 0.0, 0.0),
            r_mat=r_mat,
        )
        self.assertEqual(baseline, direct)


if __name__ == "__main__":
    unittest.main()
