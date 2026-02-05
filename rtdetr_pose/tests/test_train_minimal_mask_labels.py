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
class TestTrainMinimalMaskLabels(unittest.TestCase):
    def test_labels_from_instance_mask(self):
        mod = _load_train_minimal_module()
        record = {
            "labels": [],
            "mask": [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
            ],
            "mask_format": "instance",
            "mask_class_id": 3,
        }
        ds = mod.ManifestDataset(
            [record],
            num_queries=5,
            num_classes=10,
            image_size=4,
            seed=0,
            use_matcher=True,
            synthetic_pose=False,
            z_from_dobj=False,
            load_aux=True,
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
            jitter_dx=0.01,
            jitter_dy=0.01,
            jitter_dz=0.02,
            jitter_droll=1.0,
            jitter_dpitch=1.0,
            jitter_dyaw=2.0,
        )
        item = ds[0]
        targets = item["targets"]
        self.assertEqual(targets["gt_labels"].shape[0], 2)
        self.assertTrue((targets["gt_labels"] == 3).all())


if __name__ == "__main__":
    unittest.main()
