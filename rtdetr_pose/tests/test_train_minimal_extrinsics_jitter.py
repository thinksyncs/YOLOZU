import importlib.util
import math
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
class TestTrainMinimalExtrinsicsJitter(unittest.TestCase):
    def test_sim_extrinsics_jitter_applies(self):
        mod = _load_train_minimal_module()

        record = {
            "image_path": "",
            "labels": [{"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
            "t_gt": [[0.0, 0.0, 1.0]],
            "R_gt": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        }
        profile = {
            "intrinsics": {"dfx": 0.0, "dfy": 0.0, "dcx": 0.0, "dcy": 0.0},
            "extrinsics": {"dx": 0.05, "dy": 0.01, "dz": 0.02, "droll": 1.0, "dpitch": 2.0, "dyaw": 3.0},
            "rolling_shutter": {"enabled": False, "line_delay": 0.0},
        }

        ds = mod.ManifestDataset(
            [record],
            num_queries=5,
            num_classes=80,
            image_size=10,
            seed=42,
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
            sim_jitter_extrinsics=True,
            extrinsics_jitter=False,
            jitter_dx=0.0,
            jitter_dy=0.0,
            jitter_dz=0.0,
            jitter_droll=0.0,
            jitter_dpitch=0.0,
            jitter_dyaw=0.0,
        )

        sample = ds[0]
        t = sample["targets"]["gt_t"][0]
        r = sample["targets"]["gt_R"][0]

        jitter = mod.sample_extrinsics_jitter(profile, seed=42)
        expected_t = torch.tensor([
            0.0 + jitter["dx"],
            0.0 + jitter["dy"],
            1.0 + jitter["dz"],
        ])
        r_delta = mod._rotation_matrix_from_rpy(
            math.radians(jitter["droll"]),
            math.radians(jitter["dpitch"]),
            math.radians(jitter["dyaw"]),
        )
        expected_r = r_delta @ torch.eye(3)

        self.assertTrue(torch.allclose(t, expected_t, atol=1e-6))
        self.assertTrue(torch.allclose(r, expected_r, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
