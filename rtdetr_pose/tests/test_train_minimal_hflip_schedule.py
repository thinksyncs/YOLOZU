import importlib.util
import unittest
from pathlib import Path


def _load_train_minimal_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"
    spec = importlib.util.spec_from_file_location("rtdetr_pose_tools_train_minimal", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTrainMinimalHflipSchedule(unittest.TestCase):
    def test_linear_schedule(self):
        mod = _load_train_minimal_module()
        self.assertAlmostEqual(mod.compute_linear_schedule(0.0, 1.0, 0, 5), 0.0)
        self.assertAlmostEqual(mod.compute_linear_schedule(0.0, 1.0, 4, 5), 1.0)
        self.assertAlmostEqual(mod.compute_linear_schedule(0.0, 1.0, 2, 5), 0.5)


if __name__ == "__main__":
    unittest.main()
