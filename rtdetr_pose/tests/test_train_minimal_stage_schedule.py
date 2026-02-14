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
class TestTrainMinimalStageSchedule(unittest.TestCase):
    def test_stage_weights_progression(self):
        mod = _load_train_minimal_module()
        base = {"off": 1.0, "k": 1.0}

        weights, stage = mod.compute_stage_weights(base, global_step=0, stage_off_steps=5, stage_k_steps=7)
        self.assertEqual(stage, "offsets")
        self.assertEqual(weights["k"], 0.0)
        self.assertEqual(weights["off"], 1.0)

        weights, stage = mod.compute_stage_weights(base, global_step=5, stage_off_steps=5, stage_k_steps=7)
        self.assertEqual(stage, "k")
        self.assertEqual(weights["off"], 0.0)
        self.assertEqual(weights["k"], 1.0)

        weights, stage = mod.compute_stage_weights(base, global_step=12, stage_off_steps=5, stage_k_steps=7)
        self.assertEqual(stage, "full")
        self.assertEqual(weights["off"], 1.0)
        self.assertEqual(weights["k"], 1.0)


if __name__ == "__main__":
    unittest.main()
