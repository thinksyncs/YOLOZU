import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from yolozu.tta import TentConfig, TentRunner


@unittest.skipIf(torch is None, "torch not installed")
class TestTentRunner(unittest.TestCase):
    def test_tent_runner_step(self):
        model = nn.Sequential(nn.Linear(4, 3))
        runner = TentRunner(model, config=TentConfig(lr=1e-3))
        batch = torch.randn(2, 4)
        out = runner.adapt_step(batch)
        self.assertIn("loss_entropy", out)


if __name__ == "__main__":
    unittest.main()
