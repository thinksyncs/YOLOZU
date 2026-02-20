import sys
from pathlib import Path
import unittest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from rtdetr_pose.model import RTDETRPose
from rtdetr_pose.config import ModelConfig
from rtdetr_pose.factory import build_model


class TestModelShapes(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_forward_shapes(self):
        model = RTDETRPose(
            # RT-DETR-style: include an explicit "no-object" / background class.
            num_classes=6,
            hidden_dim=64,
            num_queries=10,
            num_decoder_layers=2,
            nhead=4,
        )
        x = torch.zeros(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out["logits"].shape, (2, 10, 6))
        self.assertEqual(out["bbox"].shape, (2, 10, 4))
        self.assertEqual(out["log_z"].shape, (2, 10, 1))
        self.assertEqual(out["rot6d"].shape, (2, 10, 6))
        self.assertEqual(out["offsets"].shape, (2, 10, 2))
        self.assertEqual(out["k_delta"].shape, (2, 4))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_factory_tiny_backbone(self):
        cfg = ModelConfig(
            num_classes=5,
            hidden_dim=64,
            num_queries=10,
            num_decoder_layers=2,
            nhead=4,
            backbone_name="tiny_cnn",
            stem_channels=16,
            backbone_channels=[32, 64, 128],
            stage_blocks=[1, 1, 1],
            depth_mode="none",
            depth_dropout=0.0,
        )
        model = build_model(cfg)
        x = torch.zeros(2, 3, 64, 64)
        out = model(x)
        # build_model() treats cfg.num_classes as foreground and adds +1 background.
        self.assertEqual(out["logits"].shape, (2, 10, 6))
        self.assertEqual(out["bbox"].shape, (2, 10, 4))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_factory_activation_backbone_head_override(self):
        cfg = ModelConfig(
            num_classes=5,
            hidden_dim=64,
            num_queries=10,
            num_decoder_layers=2,
            nhead=4,
            backbone_name="tiny_cnn",
            stem_channels=16,
            backbone_channels=[32, 64, 128],
            stage_blocks=[1, 1, 1],
            backbone_activation="gelu",
            head_activation="leakyrelu",
        )
        model = build_model(cfg)
        self.assertIsInstance(model.backbone.stem[0].act, nn.GELU)
        self.assertIsInstance(model.encoder.fpn.lateral[0].act, nn.LeakyReLU)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_factory_activation_preset_head_leakyrelu(self):
        cfg = ModelConfig(
            num_classes=5,
            hidden_dim=64,
            num_queries=10,
            num_decoder_layers=2,
            nhead=4,
            backbone_name="tiny_cnn",
            stem_channels=16,
            backbone_channels=[32, 64, 128],
            stage_blocks=[1, 1, 1],
            activation_preset="head_leakyrelu",
        )
        model = build_model(cfg)
        self.assertIsInstance(model.backbone.stem[0].act, nn.SiLU)
        self.assertIsInstance(model.encoder.fpn.lateral[0].act, nn.LeakyReLU)


if __name__ == "__main__":
    unittest.main()
