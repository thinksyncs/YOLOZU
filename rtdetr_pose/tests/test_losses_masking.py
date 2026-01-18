import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestLossesMasking(unittest.TestCase):
    def test_bbox_loss_uses_valid_mask(self):
        from rtdetr_pose.losses import Losses

        losses = Losses(weights={"cls": 0.0, "box": 1.0})

        bbox_pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]])
        bbox_gt = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [1.5, 1.0, 1.0, 1.0]]])
        labels = torch.tensor([[-1, 3]], dtype=torch.long)

        out = {"bbox": bbox_pred}
        targets = {"bbox": bbox_gt, "labels": labels}

        d = losses(out, targets)
        # Only query 1 is valid: |1-1.5| + 0 + 0 + 0 == 0.5
        self.assertAlmostEqual(float(d["loss_box"].detach().cpu()), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
