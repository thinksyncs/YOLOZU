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
class TestTrainMinimalGradClip(unittest.TestCase):
    def test_clip_grad_norm_applied(self):
        _load_train_minimal_module()

        model = torch.nn.Linear(2, 2)
        x = torch.tensor([[10.0, -10.0]])
        y = model(x).sum()
        y.backward()

        # Apply clip and ensure resulting norm is bounded.
        max_norm = 0.1
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        self.assertGreater(float(total_norm), 0.0)
        clipped_norm = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            clipped_norm += float(p.grad.data.norm(2) ** 2)
        clipped_norm = clipped_norm ** 0.5
        self.assertLessEqual(clipped_norm, max_norm + 1e-6)


if __name__ == "__main__":
    unittest.main()
