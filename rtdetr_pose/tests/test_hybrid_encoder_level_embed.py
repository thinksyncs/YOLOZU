import unittest
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "rtdetr_pose"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from rtdetr_pose.model import HybridEncoder


@unittest.skipIf(torch is None, "torch not installed")
class TestHybridEncoderLevelEmbed(unittest.TestCase):
    def test_level_embed_toggle(self):
        encoder = HybridEncoder(in_channels=(8, 16, 32), hidden_dim=8, use_level_embed=True)
        self.assertIsNotNone(encoder.level_embed)
        self.assertEqual(tuple(encoder.level_embed.shape), (3, 8))

        encoder_no = HybridEncoder(in_channels=(8, 16, 32), hidden_dim=8, use_level_embed=False)
        self.assertIsNone(encoder_no.level_embed)

    def test_forward_shapes(self):
        encoder = HybridEncoder(in_channels=(8, 16, 32), hidden_dim=8, use_level_embed=True)
        feats = [
            torch.rand(1, 8, 8, 8),
            torch.rand(1, 16, 4, 4),
            torch.rand(1, 32, 2, 2),
        ]
        pos = [
            torch.zeros(1, 64, 8),
            torch.zeros(1, 16, 8),
            torch.zeros(1, 4, 8),
        ]
        memory, fused = encoder(feats, pos)
        self.assertEqual(tuple(memory.shape), (1, 84, 8))
        self.assertEqual(len(fused), 3)


if __name__ == "__main__":
    unittest.main()
