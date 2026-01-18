import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestTrainMinimalMatcher(unittest.TestCase):
    def test_build_query_aligned_targets_runs(self):
        from rtdetr_pose.training import build_query_aligned_targets

        b, q, c = 2, 5, 4
        logits = torch.randn(b, q, c)
        bbox = torch.randn(b, q, 4)

        targets = [
            {
                "gt_labels": torch.tensor([1, 2], dtype=torch.long),
                "gt_bbox": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.2, 0.2, 0.1, 0.1]], dtype=torch.float32),
            },
            {
                "gt_labels": torch.tensor([], dtype=torch.long),
                "gt_bbox": torch.zeros((0, 4), dtype=torch.float32),
            },
        ]

        aligned = build_query_aligned_targets(logits, bbox, targets, num_queries=q)
        self.assertEqual(aligned["labels"].shape, (b, q))
        self.assertEqual(aligned["bbox"].shape, (b, q, 4))
        self.assertEqual(aligned["mask"].shape, (b, q))


if __name__ == "__main__":
    unittest.main()
