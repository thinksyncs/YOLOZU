import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestLossesTranslation(unittest.TestCase):
    def test_translation_loss_zero_when_consistent(self):
        from rtdetr_pose.losses import Losses

        b, q = 1, 2
        image_hw = torch.tensor([[32.0, 32.0]], dtype=torch.float32)
        K_gt = torch.tensor([[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)

        # bbox centers in normalized coords.
        bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.2, 0.2]]], dtype=torch.float32)
        offsets = torch.zeros((b, q, 2), dtype=torch.float32)

        # Choose z, derive t_gt from K and bbox center with offsets=0.
        z = torch.tensor([[0.4, 0.8]], dtype=torch.float32)
        u = bbox[..., 0] * image_hw[:, None, 1]
        v = bbox[..., 1] * image_hw[:, None, 0]
        fx = K_gt[:, 0, 0].unsqueeze(1)
        fy = K_gt[:, 1, 1].unsqueeze(1)
        cx = K_gt[:, 0, 2].unsqueeze(1)
        cy = K_gt[:, 1, 2].unsqueeze(1)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        t_gt = torch.stack((x, y, z), dim=-1)

        outputs = {
            "log_z": torch.log(z).unsqueeze(-1),
            "offsets": offsets,
            "k_delta": torch.zeros((b, 4), dtype=torch.float32),
        }
        targets = {
            "bbox": bbox,
            "t_gt": t_gt,
            "K_gt": K_gt,
            "image_hw": image_hw,
            "labels": torch.tensor([[1, 1]], dtype=torch.long),
        }

        losses = Losses(weights={"cls": 0.0, "box": 0.0, "z": 0.0, "rot": 0.0, "off": 0.0, "k": 0.0, "t": 1.0})
        d = losses(outputs, targets)
        self.assertIn("loss_t", d)
        self.assertLess(float(d["loss_t"].detach().cpu()), 1e-6)


if __name__ == "__main__":
    unittest.main()
