import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestLossesEmptyGT(unittest.TestCase):
    def test_cls_loss_zero_when_all_ignored(self):
        from rtdetr_pose.losses import Losses

        b, q, c = 2, 4, 5
        outputs = {"logits": torch.randn(b, q, c)}
        targets = {"labels": torch.full((b, q), -1, dtype=torch.long)}

        losses = Losses(weights={"cls": 1.0, "box": 0.0, "z": 0.0, "rot": 0.0, "off": 0.0, "k": 0.0, "t": 0.0})
        d = losses(outputs, targets)

        self.assertTrue(torch.isfinite(d["loss_cls"]))
        self.assertAlmostEqual(float(d["loss_cls"].detach().cpu()), 0.0, places=7)
        self.assertAlmostEqual(float(d["loss"].detach().cpu()), 0.0, places=7)

    def test_z_loss_zero_when_mask_empty(self):
        from rtdetr_pose.losses import Losses

        b, q = 1, 3
        outputs = {"log_z": torch.randn(b, q, 1)}
        targets = {
            "labels": torch.full((b, q), -1, dtype=torch.long),
            "z_gt": torch.ones(b, q, 1),
        }

        losses = Losses(weights={"cls": 0.0, "box": 0.0, "z": 1.0, "rot": 0.0, "off": 0.0, "k": 0.0, "t": 0.0})
        d = losses(outputs, targets)

        self.assertTrue(torch.isfinite(d["loss_z"]))
        self.assertAlmostEqual(float(d["loss_z"].detach().cpu()), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
