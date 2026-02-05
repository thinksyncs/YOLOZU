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
class TestTrainMinimalDenoise(unittest.TestCase):
    def test_apply_denoise_targets_appends(self):
        mod = _load_train_minimal_module()
        targets = [
            {
                "gt_labels": torch.tensor([1, 2], dtype=torch.long),
                "gt_bbox": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.1, 0.2]], dtype=torch.float32),
                "gt_z": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            }
        ]
        out = mod.apply_denoise_targets(
            targets,
            num_classes=5,
            denoise_count=1,
            bbox_noise=0.01,
            label_noise=0.0,
        )
        self.assertEqual(out[0]["gt_labels"].shape[0], 4)
        self.assertEqual(out[0]["gt_bbox"].shape[0], 4)
        self.assertEqual(out[0]["gt_z"].shape[0], 4)


if __name__ == "__main__":
    unittest.main()
