import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from yolozu.sdft import SdftConfig, compute_sdft_loss, kl_divergence_from_logits


class TestSDFT(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_kl_zero_when_equal(self):
        logits = torch.randn(2, 3, 5)
        loss_f = kl_divergence_from_logits(logits, logits, mode="forward")
        loss_r = kl_divergence_from_logits(logits, logits, mode="reverse")
        loss_s = kl_divergence_from_logits(logits, logits, mode="sym")
        self.assertLess(float(loss_f.detach().cpu()), 1e-7)
        self.assertLess(float(loss_r.detach().cpu()), 1e-7)
        self.assertLess(float(loss_s.detach().cpu()), 1e-7)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_compute_sdft_loss_parts(self):
        student = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
        }
        teacher = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
        }
        cfg = SdftConfig(
            weight=1.0,
            temperature=2.0,
            kl="reverse",
            keys=("logits", "bbox"),
            logits_weight=0.7,
            bbox_weight=0.3,
        )
        total, parts = compute_sdft_loss(student, teacher, cfg)
        self.assertIn("loss_sdft", parts)
        self.assertIn("loss_sdft_logits", parts)
        self.assertIn("loss_sdft_bbox", parts)
        self.assertTrue(torch.is_tensor(total))
        self.assertTrue(torch.allclose(total, parts["loss_sdft"]))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_missing_keys_zero_on_reference_device(self):
        logits = torch.randn(1, 2, 3)
        student = {"logits": logits}
        teacher = {"logits": logits.clone()}
        cfg = SdftConfig(keys=("missing",))
        total, parts = compute_sdft_loss(student, teacher, cfg)
        self.assertEqual(total.device, logits.device)
        self.assertEqual(total.dtype, logits.dtype)
        self.assertIn("loss_sdft", parts)
        self.assertEqual(float(total.detach().cpu()), 0.0)


if __name__ == "__main__":
    unittest.main()

