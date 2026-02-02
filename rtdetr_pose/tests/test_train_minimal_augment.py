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
class TestTrainMinimalAugment(unittest.TestCase):
    def test_multiscale_and_hflip_updates_targets(self):
        mod = _load_train_minimal_module()

        record = {
            "image_path": "",  # unused (synthetic)
            "labels": [{"class_id": 0, "bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.2}}],
            "offsets": [[2.0, 1.0]],
        }

        ds = mod.ManifestDataset(
            [record],
            num_queries=5,
            num_classes=80,
            image_size=10,
            seed=123,
            use_matcher=True,
            synthetic_pose=False,
            z_from_dobj=False,
            load_aux=False,
            real_images=False,
            multiscale=True,
            scale_min=0.5,
            scale_max=0.5,
            hflip_prob=1.0,
            intrinsics_jitter=False,
            jitter_dfx=0.02,
            jitter_dfy=0.02,
            jitter_dcx=4.0,
            jitter_dcy=4.0,
            sim_jitter=False,
            sim_jitter_profile=None,
        )

        sample = ds[0]
        image = sample["image"]
        targets = sample["targets"]

        self.assertEqual(tuple(image.shape[-2:]), (5, 5))
        self.assertTrue(torch.allclose(targets["image_hw"], torch.tensor([5.0, 5.0])))
        # cx should be flipped: 1 - 0.2 = 0.8
        self.assertAlmostEqual(float(targets["gt_bbox"][0][0]), 0.8, places=6)
        # offsets should be scaled by 0.5 and x negated due to flip
        self.assertAlmostEqual(float(targets["gt_offsets"][0][0]), -1.0, places=6)
        self.assertAlmostEqual(float(targets["gt_offsets"][0][1]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
