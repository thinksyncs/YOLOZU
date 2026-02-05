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

from yolozu.adapter import DummyAdapter, ModelAdapter
from yolozu.tta.config import TTTConfig
from yolozu.tta.integration import run_ttt


@unittest.skipIf(torch is None, "torch not installed")
class TestTTTIntegration(unittest.TestCase):
    def test_run_ttt_tent(self):
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
                yield torch.randn(2, 4)

        adapter = Adapter()
        records = [{"image": "a.jpg"}, {"image": "b.jpg"}]
        report = run_ttt(adapter, records, config=TTTConfig(enabled=True, method="tent", steps=2, lr=1e-3))
        self.assertTrue(report.enabled)
        self.assertEqual(report.method, "tent")
        self.assertEqual(report.steps_run, 2)
        self.assertEqual(len(report.losses), 2)
        self.assertIsInstance(report.updated_param_count, int)
        self.assertGreater(report.updated_param_count or 0, 0)

    def test_run_ttt_mim(self):
        class ReconModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)

            def forward(self, x):
                return self.conv(x)

        class Adapter(ModelAdapter):
            def __init__(self):
                self._model = ReconModel()

            def predict(self, records):
                return [{"image": r["image"], "detections": []} for r in records]

            def supports_ttt(self) -> bool:
                return True

            def get_model(self):
                return self._model

            def build_loader(self, records, *, batch_size: int = 1):
                yield torch.rand(1, 3, 8, 8)

        adapter = Adapter()
        records = [{"image": "a.jpg"}]
        report = run_ttt(
            adapter,
            records,
            config=TTTConfig(
                enabled=True,
                method="mim",
                steps=2,
                lr=1e-3,
                mim_mask_prob=0.5,
                mim_patch_size=2,
            ),
        )
        self.assertTrue(report.enabled)
        self.assertEqual(report.method, "mim")
        self.assertEqual(report.steps_run, 2)
        self.assertEqual(len(report.losses), 2)
        self.assertIsInstance(report.updated_param_count, int)
        self.assertGreater(report.updated_param_count or 0, 0)

    def test_run_ttt_unsupported_adapter_errors(self):
        adapter = DummyAdapter()
        records = [{"image": "a.jpg"}]
        with self.assertRaises(RuntimeError):
            run_ttt(adapter, records, config=TTTConfig(enabled=True, method="tent", steps=1))


if __name__ == "__main__":
    unittest.main()
