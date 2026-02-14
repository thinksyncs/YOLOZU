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
class TestTrainMinimalCostSchedule(unittest.TestCase):
    def test_stage_costs_progression(self):
        mod = _load_train_minimal_module()
        base = {"cost_z": 1.0, "cost_rot": 2.0, "cost_t": 3.0}

        costs = mod.compute_stage_costs(
            base,
            global_step=0,
            cost_z_start_step=5,
            cost_rot_start_step=10,
            cost_t_start_step=15,
        )
        self.assertEqual(costs["cost_z"], 0.0)
        self.assertEqual(costs["cost_rot"], 0.0)
        self.assertEqual(costs["cost_t"], 0.0)

        costs = mod.compute_stage_costs(
            base,
            global_step=5,
            cost_z_start_step=5,
            cost_rot_start_step=10,
            cost_t_start_step=15,
        )
        self.assertEqual(costs["cost_z"], 1.0)
        self.assertEqual(costs["cost_rot"], 0.0)
        self.assertEqual(costs["cost_t"], 0.0)

        costs = mod.compute_stage_costs(
            base,
            global_step=10,
            cost_z_start_step=5,
            cost_rot_start_step=10,
            cost_t_start_step=15,
        )
        self.assertEqual(costs["cost_z"], 1.0)
        self.assertEqual(costs["cost_rot"], 2.0)
        self.assertEqual(costs["cost_t"], 0.0)

        costs = mod.compute_stage_costs(
            base,
            global_step=15,
            cost_z_start_step=5,
            cost_rot_start_step=10,
            cost_t_start_step=15,
        )
        self.assertEqual(costs["cost_z"], 1.0)
        self.assertEqual(costs["cost_rot"], 2.0)
        self.assertEqual(costs["cost_t"], 3.0)


if __name__ == "__main__":
    unittest.main()
