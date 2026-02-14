import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "rtdetr_pose"))


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

    def test_depth_loss_respects_mask_availability(self):
        from rtdetr_pose.losses import Losses

        losses = Losses(weights={"cls": 0.0, "box": 0.0, "z": 1.0})

        log_z_pred = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        z_gt = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        d_obj_mask = torch.tensor([[False, False]], dtype=torch.bool)

        out = {"log_z": log_z_pred}
        targets = {"z_gt": z_gt, "D_obj_mask": d_obj_mask, "mask": torch.tensor([[True, True]])}

        d = losses(out, targets)
        self.assertAlmostEqual(float(d["loss_z"].detach().cpu()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
