import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rtdetr_pose"))

from rtdetr_pose.task_aligner import align_losses_uncertainty, uncertainty_weighted


class TestTaskAligner(unittest.TestCase):
    def test_uncertainty_weighted_identity_at_zero(self):
        self.assertAlmostEqual(uncertainty_weighted(3.0, 0.0), 3.0, places=7)

    def test_uncertainty_weighted_downweights_large_loss(self):
        # Large positive log_sigma reduces exp(-s) multiplier, so large raw loss is downweighted.
        raw = 10.0
        aligned = uncertainty_weighted(raw, 2.0)
        self.assertLess(aligned, raw)

    def test_align_losses_uncertainty_adds_aligned_keys(self):
        aligned = align_losses_uncertainty(losses={"loss_z": 10.0, "loss_rot": 1.0}, log_sigmas={"z": 2.0, "rot": 0.0})
        self.assertIn("loss_z_aligned", aligned)
        self.assertIn("loss_rot_aligned", aligned)
        self.assertLess(aligned["loss_z_aligned"], 10.0)


if __name__ == "__main__":
    unittest.main()
