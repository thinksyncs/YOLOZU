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

from yolozu.tta.ttt_mim import (
    apply_mask,
    filter_parameters,
    generate_block_mask,
    run_ttt_mim,
)


@unittest.skipIf(torch is None, "torch not installed")
class TestTTTMIM(unittest.TestCase):
    def test_generate_block_mask_full(self):
        mask = generate_block_mask(4, 4, patch_size=2, mask_prob=1.0)
        self.assertTrue(mask.all())

    def test_apply_mask(self):
        x = torch.ones(1, 3, 4, 4)
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[0:2, 0:2] = True
        masked = apply_mask(x, mask, mask_value=0.0)
        self.assertTrue(torch.allclose(masked[0, :, 0, 0], torch.zeros(3)))
        self.assertTrue(torch.allclose(masked[0, :, 3, 3], torch.ones(3)))

    def test_filter_parameters_include_exclude(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Conv2d(4, 2, 1))
        params = filter_parameters(model.named_parameters(), include=["0"], exclude=["bias"])
        self.assertTrue(params)
        for p in params:
            self.assertTrue(p.requires_grad)

    def test_run_ttt_mim(self):
        torch.manual_seed(0)
        model = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        x = torch.rand(1, 3, 8, 8)
        res = run_ttt_mim(model, x, steps=2, mask_prob=0.5, patch_size=2)
        self.assertEqual(len(res.losses), 2)
        self.assertGreaterEqual(res.mask_ratio, 0.0)
        self.assertLessEqual(res.mask_ratio, 1.0)
        self.assertGreater(res.updated_param_count, 0)

    def test_update_filter_norm_only(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
        x = torch.rand(1, 3, 4, 4)
        res = run_ttt_mim(model, x, steps=1, mask_prob=0.5, patch_size=2, update_filter="norm_only")
        expected = sum(p.numel() for p in model[1].parameters())
        self.assertEqual(res.updated_param_count, expected)

    def test_update_filter_adapter_only(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = nn.Conv2d(3, 3, 1)
                self.head = nn.Conv2d(3, 3, 1)

            def forward(self, x):
                return self.adapter(x)

        model = Model()
        x = torch.rand(1, 3, 4, 4)
        res = run_ttt_mim(model, x, steps=1, mask_prob=0.5, patch_size=2, update_filter="adapter_only")
        expected = sum(p.numel() for p in model.adapter.parameters())
        self.assertEqual(res.updated_param_count, expected)


if __name__ == "__main__":
    unittest.main()
