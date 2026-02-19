import types
import unittest

from yolozu.tta.presets import apply_ttt_preset_args


class TestTTTPresets(unittest.TestCase):
    def test_auto_applies_safe_preset_for_tent_when_defaultish(self):
        args = types.SimpleNamespace(
            ttt=True,
            ttt_preset=None,
            ttt_method="tent",
            ttt_steps=1,
            ttt_batch_size=1,
            ttt_lr=1e-4,
            ttt_update_filter="all",
            ttt_max_batches=1,
            ttt_max_grad_norm=None,
            ttt_max_update_norm=None,
            ttt_max_total_update_norm=None,
            ttt_max_loss_ratio=None,
            ttt_max_loss_increase=None,
        )

        apply_ttt_preset_args(args)

        self.assertEqual(args.ttt_preset, "safe")
        self.assertEqual(args.ttt_method, "tent")
        self.assertEqual(args.ttt_update_filter, "norm_only")
        self.assertEqual(args.ttt_max_grad_norm, 1.0)
        self.assertEqual(args.ttt_max_update_norm, 1.0)
        self.assertEqual(args.ttt_max_total_update_norm, 1.0)
        self.assertEqual(args.ttt_max_loss_ratio, 3.0)

    def test_auto_applies_safe_preset_for_mim_when_defaultish(self):
        args = types.SimpleNamespace(
            ttt=True,
            ttt_preset=None,
            ttt_method="mim",
            ttt_steps=1,
            ttt_batch_size=1,
            ttt_lr=1e-4,
            ttt_update_filter="all",
            ttt_max_batches=1,
            ttt_max_grad_norm=None,
            ttt_max_update_norm=None,
            ttt_max_total_update_norm=None,
            ttt_max_loss_ratio=None,
            ttt_max_loss_increase=None,
        )

        apply_ttt_preset_args(args)

        self.assertEqual(args.ttt_preset, "mim_safe")
        self.assertEqual(args.ttt_method, "mim")
        self.assertEqual(args.ttt_update_filter, "adapter_only")
        self.assertEqual(args.ttt_max_grad_norm, 5.0)
        self.assertEqual(args.ttt_max_update_norm, 5.0)
        self.assertEqual(args.ttt_max_total_update_norm, 5.0)
        self.assertEqual(args.ttt_max_loss_ratio, 3.0)

    def test_fills_safety_without_overriding_core_when_customized(self):
        args = types.SimpleNamespace(
            ttt=True,
            ttt_preset=None,
            ttt_method="tent",
            ttt_steps=5,
            ttt_batch_size=1,
            ttt_lr=1e-4,
            ttt_update_filter="adapter_only",
            ttt_max_batches=2,
            ttt_max_grad_norm=None,
            ttt_max_update_norm=None,
            ttt_max_total_update_norm=None,
            ttt_max_loss_ratio=None,
            ttt_max_loss_increase=None,
        )

        apply_ttt_preset_args(args)

        self.assertIsNone(args.ttt_preset)
        self.assertEqual(args.ttt_method, "tent")
        self.assertEqual(args.ttt_steps, 5)
        self.assertEqual(args.ttt_update_filter, "adapter_only")
        self.assertEqual(args.ttt_max_batches, 2)
        self.assertEqual(args.ttt_max_grad_norm, 5.0)
        self.assertEqual(args.ttt_max_update_norm, 5.0)
        self.assertEqual(args.ttt_max_total_update_norm, 5.0)
        self.assertEqual(args.ttt_max_loss_ratio, 3.0)

    def test_auto_applies_cotta_safe_preset_when_defaultish(self):
        args = types.SimpleNamespace(
            ttt=True,
            ttt_preset=None,
            ttt_method="cotta",
            ttt_steps=1,
            ttt_batch_size=1,
            ttt_lr=1e-4,
            ttt_update_filter="all",
            ttt_max_batches=1,
            ttt_max_grad_norm=None,
            ttt_max_update_norm=None,
            ttt_max_total_update_norm=None,
            ttt_max_loss_ratio=None,
            ttt_max_loss_increase=None,
        )

        apply_ttt_preset_args(args)

        self.assertEqual(args.ttt_preset, "cotta_safe")
        self.assertEqual(args.ttt_method, "cotta")
        self.assertEqual(args.ttt_update_filter, "lora_norm_only")
        self.assertEqual(args.ttt_max_grad_norm, 1.0)
        self.assertEqual(args.ttt_max_update_norm, 1.0)
        self.assertEqual(args.ttt_max_total_update_norm, 1.0)
        self.assertEqual(args.ttt_max_loss_ratio, 3.0)

    def test_explicit_cotta_safe_preset_overrides_core(self):
        args = types.SimpleNamespace(
            ttt=True,
            ttt_preset="cotta_safe",
            ttt_method="tent",
            ttt_steps=5,
            ttt_batch_size=2,
            ttt_lr=1e-3,
            ttt_update_filter="all",
            ttt_max_batches=4,
            ttt_max_grad_norm=None,
            ttt_max_update_norm=None,
            ttt_max_total_update_norm=None,
            ttt_max_loss_ratio=None,
            ttt_max_loss_increase=None,
        )

        apply_ttt_preset_args(args)

        self.assertEqual(args.ttt_method, "cotta")
        self.assertEqual(args.ttt_steps, 1)
        self.assertEqual(args.ttt_batch_size, 1)
        self.assertEqual(args.ttt_lr, 1e-4)
        self.assertEqual(args.ttt_update_filter, "lora_norm_only")
        self.assertEqual(args.ttt_max_batches, 1)

    def test_auto_applies_eata_safe_preset_when_defaultish(self):
        args = types.SimpleNamespace(
            ttt=True,
            ttt_preset=None,
            ttt_method="eata",
            ttt_steps=1,
            ttt_batch_size=1,
            ttt_lr=1e-4,
            ttt_update_filter="all",
            ttt_max_batches=1,
            ttt_max_grad_norm=None,
            ttt_max_update_norm=None,
            ttt_max_total_update_norm=None,
            ttt_max_loss_ratio=None,
            ttt_max_loss_increase=None,
        )

        apply_ttt_preset_args(args)

        self.assertEqual(args.ttt_preset, "eata_safe")
        self.assertEqual(args.ttt_method, "eata")
        self.assertEqual(args.ttt_update_filter, "lora_norm_only")
        self.assertEqual(args.ttt_max_grad_norm, 1.0)
        self.assertEqual(args.ttt_max_update_norm, 1.0)
        self.assertEqual(args.ttt_max_total_update_norm, 1.0)
        self.assertEqual(args.ttt_max_loss_ratio, 3.0)


if __name__ == "__main__":
    unittest.main()
