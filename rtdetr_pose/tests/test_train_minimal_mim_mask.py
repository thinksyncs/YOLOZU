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
class TestTrainMinimalMimMask(unittest.TestCase):
    def test_mim_mask_applies_all_patches(self):
        mod = _load_train_minimal_module()

        record = {
            "image_path": "",
            "labels": [{"class_id": 0, "bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.2}}],
        }

        ds = mod.ManifestDataset(
            [record],
            num_queries=5,
            num_classes=80,
            image_size=4,
            seed=0,
            use_matcher=False,
            synthetic_pose=False,
            z_from_dobj=False,
            load_aux=False,
            real_images=False,
            multiscale=False,
            scale_min=1.0,
            scale_max=1.0,
            hflip_prob=0.0,
            intrinsics_jitter=False,
            jitter_dfx=0.0,
            jitter_dfy=0.0,
            jitter_dcx=0.0,
            jitter_dcy=0.0,
            sim_jitter=False,
            sim_jitter_profile=None,
            sim_jitter_extrinsics=False,
            extrinsics_jitter=False,
            jitter_dx=0.0,
            jitter_dy=0.0,
            jitter_dz=0.0,
            jitter_droll=0.0,
            jitter_dpitch=0.0,
            jitter_dyaw=0.0,
            mim_mask_prob=1.0,
            mim_mask_size=2,
            mim_mask_value=0.0,
        )

        sample = ds[0]
        image = sample["image"]
        self.assertAlmostEqual(float(sample["mim_mask_ratio"]), 1.0, places=6)
        self.assertTrue(torch.allclose(image, torch.zeros_like(image)))


if __name__ == "__main__":
    unittest.main()
