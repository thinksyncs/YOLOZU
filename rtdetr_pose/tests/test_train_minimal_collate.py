import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _load_train_minimal_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"
    spec = importlib.util.spec_from_file_location("rtdetr_pose_tools_train_minimal", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipIf(torch is None, "torch not installed")
class TestTrainMinimalCollate(unittest.TestCase):
    def test_collate_pads_variable_instances(self):
        mod = _load_train_minimal_module()

        item1 = {
            "image": torch.zeros(3, 4, 4),
            "targets": {
                "gt_labels": torch.tensor([1, 2], dtype=torch.long),
                "gt_bbox": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.1, 0.2]], dtype=torch.float32),
                "gt_z": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
                "gt_R": torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1),
            },
        }
        item2 = {
            "image": torch.zeros(3, 4, 4),
            "targets": {
                "gt_labels": torch.empty((0,), dtype=torch.long),
                "gt_bbox": torch.empty((0, 4), dtype=torch.float32),
                "gt_z": torch.empty((0, 1), dtype=torch.float32),
                "gt_R": torch.empty((0, 3, 3), dtype=torch.float32),
            },
        }

        images, targets = mod.collate([item1, item2])
        self.assertEqual(tuple(images.shape), (2, 3, 4, 4))
        self.assertIsInstance(targets, dict)
        self.assertIn("per_sample", targets)
        self.assertIn("padded", targets)

        padded = targets["padded"]
        self.assertEqual(tuple(padded["gt_labels"].shape), (2, 2))
        self.assertEqual(tuple(padded["gt_bbox"].shape), (2, 2, 4))
        self.assertEqual(tuple(padded["gt_z"].shape), (2, 2, 1))
        self.assertEqual(tuple(padded["gt_R"].shape), (2, 2, 3, 3))
        self.assertTrue(torch.equal(padded["gt_count"], torch.tensor([2, 0], dtype=torch.long)))
        self.assertTrue(torch.equal(padded["gt_mask"][0], torch.tensor([True, True])))
        self.assertTrue(torch.equal(padded["gt_mask"][1], torch.tensor([False, False])))
        self.assertTrue(torch.equal(padded["gt_labels"][1], torch.tensor([-1, -1], dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
