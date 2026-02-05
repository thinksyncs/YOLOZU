import sys
from pathlib import Path
import unittest


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestLoRA(unittest.TestCase):
    def test_apply_lora_head_replaces_linears(self):
        from rtdetr_pose.model import RTDETRPose
        from rtdetr_pose.lora import LoRALinear, apply_lora

        model = RTDETRPose(num_classes=3, hidden_dim=16, num_queries=2, num_decoder_layers=1, nhead=2)
        replaced = apply_lora(model, r=4, target="head")
        self.assertGreaterEqual(replaced, 1)
        self.assertIsInstance(model.head.cls, LoRALinear)
        self.assertIsInstance(model.head.box, LoRALinear)
        self.assertIsInstance(model.head.log_z, LoRALinear)
        self.assertIsInstance(model.head.rot6d, LoRALinear)

        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        self.assertIn("logits", out)
        self.assertIn("bbox", out)

    def test_freeze_base_trains_only_lora(self):
        from rtdetr_pose.model import RTDETRPose
        from rtdetr_pose.lora import LoRALinear, apply_lora, count_trainable_params, mark_only_lora_as_trainable

        model = RTDETRPose(num_classes=3, hidden_dim=16, num_queries=2, num_decoder_layers=1, nhead=2)
        apply_lora(model, r=2, target="head")
        info = mark_only_lora_as_trainable(model)
        self.assertGreater(info["lora_params"], 0)
        self.assertEqual(info["bias_params"], 0)

        # Ensure trainable params match LoRA params only.
        self.assertEqual(count_trainable_params(model), info["lora_params"])

        # Ensure a representative base param is frozen.
        some_base = model.backbone.stem[0].conv.weight
        self.assertFalse(bool(some_base.requires_grad))

        # Ensure LoRA params are trainable.
        self.assertIsInstance(model.head.cls, LoRALinear)
        self.assertTrue(bool(model.head.cls.lora_A.requires_grad))
        self.assertTrue(bool(model.head.cls.lora_B.requires_grad))
