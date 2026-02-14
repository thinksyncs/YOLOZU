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
class TestTrainMinimalMimTeacher(unittest.TestCase):
    def test_collate_includes_image_raw_and_ratio(self):
        mod = _load_train_minimal_module()
        item = {
            "image": torch.zeros(3, 4, 4),
            "image_raw": torch.ones(3, 4, 4),
            "mim_mask_ratio": 0.5,
            "targets": {"gt_labels": torch.empty((0,), dtype=torch.long), "gt_bbox": torch.empty((0, 4))},
        }
        images, targets = mod.collate([item])
        self.assertEqual(tuple(images.shape), (1, 3, 4, 4))
        self.assertIn("image_raw", targets)
        self.assertIn("mim_mask_ratio", targets)
        self.assertTrue(torch.allclose(targets["image_raw"], torch.ones(1, 3, 4, 4)))
        self.assertTrue(torch.allclose(targets["mim_mask_ratio"], torch.tensor([0.5])))


if __name__ == "__main__":
    unittest.main()
