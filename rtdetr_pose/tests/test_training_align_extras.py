import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestTrainingAlignExtras(unittest.TestCase):
    def test_aligns_z_and_R_when_present(self):
        from rtdetr_pose.training import build_query_aligned_targets

        b, q, c = 1, 4, 3
        logits = torch.randn(b, q, c)
        bbox = torch.randn(b, q, 4)
        log_z = torch.randn(b, q, 1)
        rot6d = torch.randn(b, q, 6)

        gt_labels = torch.tensor([1, 2], dtype=torch.long)
        gt_bbox = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.2, 0.2, 0.1, 0.1]], dtype=torch.float32)
        gt_z = torch.tensor([[0.4], [0.8]], dtype=torch.float32)
        gt_R = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
        gt_t = torch.tensor([[0.0, 0.0, 0.4], [0.1, 0.0, 0.8]], dtype=torch.float32)
        K_gt = torch.tensor([[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        image_hw = torch.tensor([32.0, 32.0], dtype=torch.float32)

        targets = [
            {
                "gt_labels": gt_labels,
                "gt_bbox": gt_bbox,
                "gt_z": gt_z,
                "gt_R": gt_R,
                "gt_t": gt_t,
                "K_gt": K_gt,
                "image_hw": image_hw,
            }
        ]

        aligned = build_query_aligned_targets(
            logits,
            bbox,
            targets,
            num_queries=q,
            log_z_pred=log_z,
            rot6d_pred=rot6d,
            cost_z=1.0,
            cost_rot=1.0,
            offsets_pred=torch.zeros(b, q, 2),
            k_delta=torch.zeros(b, 4),
            cost_t=1.0,
        )

        self.assertEqual(aligned["z_gt"].shape, (b, q, 1))
        self.assertEqual(aligned["R_gt"].shape, (b, q, 3, 3))
        self.assertEqual(aligned["mask"].shape, (b, q))
        self.assertEqual(aligned["t_gt"].shape, (b, q, 3))
        self.assertEqual(aligned["K_gt"].shape, (b, 3, 3))
        self.assertEqual(aligned["image_hw"].shape, (b, 2))
        self.assertEqual(aligned["K_mask"].shape, (b,))
        self.assertEqual(aligned["t_mask"].shape, (b, q))
        self.assertTrue(bool(aligned["K_mask"].all()))
        self.assertTrue(bool(aligned["t_mask"].any()))


if __name__ == "__main__":
    unittest.main()
