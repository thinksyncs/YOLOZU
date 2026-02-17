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
            sim_jitter_extrinsics=False,
            extrinsics_jitter=False,
            jitter_dx=0.01,
            jitter_dy=0.01,
            jitter_dz=0.02,
            jitter_droll=1.0,
            jitter_dpitch=1.0,
            jitter_dyaw=2.0,
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

    def test_photometric_augment_is_deterministic_and_clamped(self):
        mod = _load_train_minimal_module()

        record = {"image_path": "", "labels": [{"class_id": 0, "bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.2}}]}

        ds_base = mod.ManifestDataset([record], image_size=32, seed=123, real_images=False)
        base = ds_base[0]["image"]

        ds_aug = mod.ManifestDataset(
            [record],
            image_size=32,
            seed=123,
            real_images=False,
            hsv_h=0.02,
            hsv_s=0.2,
            hsv_v=0.2,
            hsv_prob=1.0,
            gaussian_noise_std=0.05,
            gaussian_noise_prob=1.0,
            blur_prob=1.0,
            blur_sigma=0.8,
            blur_kernel=3,
        )
        img1 = ds_aug[0]["image"]
        img2 = ds_aug[0]["image"]

        self.assertTrue(torch.all(torch.isfinite(img1)).item())
        self.assertGreaterEqual(float(img1.min().item()), 0.0)
        self.assertLessEqual(float(img1.max().item()), 1.0)
        self.assertFalse(torch.allclose(base, img1), "expected photometric aug to change pixels")
        self.assertTrue(torch.allclose(img1, img2), "expected deterministic per-index augmentation")

    def test_grayscale_aug_makes_channels_equal(self):
        mod = _load_train_minimal_module()

        record = {"image_path": "", "labels": [{"class_id": 0, "bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.2}}]}

        ds = mod.ManifestDataset([record], image_size=16, seed=7, real_images=False, gray_prob=1.0)
        img = ds[0]["image"]
        self.assertTrue(torch.allclose(img[0], img[1]))
        self.assertTrue(torch.allclose(img[1], img[2]))


if __name__ == "__main__":
    unittest.main()
