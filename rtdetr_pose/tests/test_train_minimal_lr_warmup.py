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


class TestTrainMinimalWarmup(unittest.TestCase):
    def test_compute_warmup_lr(self):
        mod = _load_train_minimal_module()

        base = 1e-3
        init = 1e-5
        warmup = 10

        self.assertEqual(mod.compute_warmup_lr(base, 0, warmup, init), init)
        self.assertEqual(mod.compute_warmup_lr(base, 10, warmup, init), base)
        mid = mod.compute_warmup_lr(base, 5, warmup, init)
        self.assertAlmostEqual(mid, init + (base - init) * 0.5, places=8)

        self.assertEqual(mod.compute_warmup_lr(base, 3, 0, init), base)


if __name__ == "__main__":
    unittest.main()
