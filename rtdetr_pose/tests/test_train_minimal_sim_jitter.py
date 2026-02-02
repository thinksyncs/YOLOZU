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
class TestTrainMinimalSimJitter(unittest.TestCase):
    def test_sim_jitter_profile_applied(self):
        mod = _load_train_minimal_module()

        record = {
            "image_path": "",
            "labels": [{"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
            "K_gt": [[10.0, 0.0, 5.0], [0.0, 12.0, 6.0], [0.0, 0.0, 1.0]],
        }
        profile = {
            "intrinsics": {"dfx": 0.1, "dfy": 0.2, "dcx": 1.5, "dcy": 2.5},
            "extrinsics": {},
            "rolling_shutter": {"enabled": False, "line_delay": 0.0},
        }

        ds = mod.ManifestDataset(
            [record],
            num_queries=5,
            num_classes=80,
            image_size=10,
            seed=321,
            use_matcher=True,
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
            sim_jitter=True,
            sim_jitter_profile=profile,
        )

        sample = ds[0]
        k = sample["targets"]["K_gt"]
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])

        self.assertGreaterEqual(fx, 10.0 * (1.0 - 0.1))
        self.assertLessEqual(fx, 10.0 * (1.0 + 0.1))
        self.assertGreaterEqual(fy, 12.0 * (1.0 - 0.2))
        self.assertLessEqual(fy, 12.0 * (1.0 + 0.2))
        self.assertGreaterEqual(cx, 5.0 - 1.5)
        self.assertLessEqual(cx, 5.0 + 1.5)
        self.assertGreaterEqual(cy, 6.0 - 2.5)
        self.assertLessEqual(cy, 6.0 + 2.5)


if __name__ == "__main__":
    unittest.main()
