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

from yolozu.adapter import ModelAdapter
from yolozu.tta.config import TTTConfig
from yolozu.tta.integration import run_ttt


@unittest.skipIf(torch is None, "torch not installed")
class TestTTTSafety(unittest.TestCase):
    def test_guard_max_update_norm_rolls_back(self):
        class Adapter(ModelAdapter):
            def __init__(self):
                self._model = nn.Sequential(nn.Linear(4, 3))

            def predict(self, records):
                return [{"image": r["image"], "detections": []} for r in records]

            def supports_ttt(self) -> bool:
                return True

            def get_model(self):
                return self._model

            def build_loader(self, records, *, batch_size: int = 1):
                _ = records, batch_size
                yield torch.randn(2, 4)

        adapter = Adapter()
        baseline = [p.detach().clone() for p in adapter.get_model().parameters()]

        report = run_ttt(
            adapter,
            [{"image": "a.jpg"}],
            config=TTTConfig(enabled=True, method="tent", steps=1, lr=1e-3, max_update_norm=0.0),
        )
        self.assertTrue(report.stopped_early)
        self.assertEqual(report.stop_reason, "max_update_norm_exceeded")
        self.assertEqual(report.steps_run, 0)
        self.assertEqual(report.losses, [])

        after = [p.detach() for p in adapter.get_model().parameters()]
        for a, b in zip(after, baseline):
            self.assertTrue(torch.equal(a, b))

        self.assertIsInstance(report.step_metrics, list)
        self.assertEqual(len(report.step_metrics or []), 1)
        self.assertTrue(bool(report.step_metrics[0].get("rolled_back")))


if __name__ == "__main__":
    unittest.main()

