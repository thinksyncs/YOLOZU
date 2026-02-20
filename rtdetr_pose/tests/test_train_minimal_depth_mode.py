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
class TestDepthMode(unittest.TestCase):
    def test_depth_none_noop_forward(self):
        from rtdetr_pose.model import RTDETRPose

        torch.manual_seed(0)
        model = RTDETRPose(num_classes=6, hidden_dim=64, num_queries=10, num_decoder_layers=2, nhead=4, depth_mode="none")
        x = torch.zeros(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(tuple(out["logits"].shape), (2, 10, 6))
        self.assertEqual(tuple(out["bbox"].shape), (2, 10, 4))

    def test_sidecar_mixed_depth_collate(self):
        mod = _load_train_minimal_module()

        records = [
            {
                "image_path": "",
                "depth": [[0.5, 0.3], [0.2, 0.1]],
                "labels": [{"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
            },
            {
                "image_path": "",
                "labels": [{"class_id": 1, "bbox": {"cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}}],
            },
        ]

        ds = mod.ManifestDataset(
            records,
            num_queries=5,
            num_classes=80,
            image_size=2,
            seed=0,
            use_matcher=True,
            synthetic_pose=False,
            z_from_dobj=False,
            load_aux=False,
            depth_mode="sidecar",
            depth_unit="relative",
            depth_scale=1.0,
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
        )

        b0 = ds[0]
        b1 = ds[1]
        self.assertIn("depth", b0)
        self.assertTrue(bool(b0.get("depth_valid", False)))
        self.assertFalse(bool(b1.get("depth_valid", True)))

        images, collated = mod.collate([b0, b1])
        self.assertEqual(tuple(images.shape), (2, 3, 2, 2))
        self.assertIn("depth", collated)
        self.assertIn("depth_valid", collated)
        self.assertEqual(tuple(collated["depth"].shape), (2, 1, 2, 2))
        self.assertEqual(collated["depth_valid"].dtype, torch.bool)
        self.assertEqual(collated["depth_valid"].tolist(), [True, False])

    def test_fuse_mid_forward_with_mixed_validity(self):
        from rtdetr_pose.model import RTDETRPose

        torch.manual_seed(0)
        model = RTDETRPose(
            num_classes=6,
            hidden_dim=64,
            num_queries=10,
            num_decoder_layers=2,
            nhead=4,
            depth_mode="fuse_mid",
            depth_dropout=0.0,
        )
        x = torch.zeros(2, 3, 64, 64)
        depth = torch.ones(2, 1, 64, 64)
        depth[1].fill_(float("nan"))
        depth_valid = torch.tensor([True, False], dtype=torch.bool)

        out = model(x, depth=depth, depth_valid=depth_valid)
        self.assertEqual(tuple(out["logits"].shape), (2, 10, 6))
        self.assertEqual(tuple(out["bbox"].shape), (2, 10, 4))


if __name__ == "__main__":
    unittest.main()
