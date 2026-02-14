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
class TestTrainMinimalMimSchedule(unittest.TestCase):
    def test_compute_mim_schedule(self):
        mod = _load_train_minimal_module()
        mask, weight = mod.compute_mim_schedule(
            step=0,
            total_steps=10,
            mask_start=0.1,
            mask_end=0.5,
            weight_start=0.0,
            weight_end=1.0,
            default_mask=0.2,
            default_weight=0.3,
        )
        self.assertAlmostEqual(mask, 0.1, places=6)
        self.assertAlmostEqual(weight, 0.0, places=6)

        mask, weight = mod.compute_mim_schedule(
            step=9,
            total_steps=10,
            mask_start=0.1,
            mask_end=0.5,
            weight_start=0.0,
            weight_end=1.0,
            default_mask=0.2,
            default_weight=0.3,
        )
        self.assertAlmostEqual(mask, 0.5, places=6)
        self.assertAlmostEqual(weight, 1.0, places=6)

        mask, weight = mod.compute_mim_schedule(
            step=1,
            total_steps=0,
            mask_start=0.1,
            mask_end=0.5,
            weight_start=0.0,
            weight_end=1.0,
            default_mask=0.2,
            default_weight=0.3,
        )
        self.assertAlmostEqual(mask, 0.2, places=6)
        self.assertAlmostEqual(weight, 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
