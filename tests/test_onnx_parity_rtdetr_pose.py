import tempfile
import unittest

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "rtdetr_pose"))


def _has_torch():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _has_onnxruntime():
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipIf(not _has_torch(), "torch not installed")
@unittest.skipIf(not _has_onnxruntime(), "onnxruntime not installed")
class TestOnnxParityRTDETRPose(unittest.TestCase):
    def test_pytorch_vs_onnxruntime_outputs_close(self):
        import numpy as np
        import torch

        from rtdetr_pose.export import export_onnx
        from rtdetr_pose.model import RTDETRPose

        torch.manual_seed(0)
        np.random.seed(0)

        model = RTDETRPose(
            num_classes=3,
            hidden_dim=64,
            num_queries=10,
            use_uncertainty=False,
            stem_channels=16,
            backbone_channels=(32, 64, 128),
            stage_blocks=(1, 1, 1),
            num_encoder_layers=0,
            num_decoder_layers=1,
            nhead=8,
        ).eval()

        # Small deterministic input.
        x = torch.linspace(0.0, 1.0, steps=3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)

        with torch.no_grad():
            pt_out = model(x)

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = f"{tmp}/model.onnx"
            export_onnx(model, x, onnx_path, opset_version=17)

            import onnxruntime as ort

            sess = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"],
            )

            ort_outs = sess.run(
                ["logits", "bbox", "log_z", "rot6d", "offsets", "k_delta"],
                {"images": x.cpu().numpy()},
            )

        keys = ["logits", "bbox", "log_z", "rot6d", "offsets", "k_delta"]
        pt_arrays = [pt_out[k].detach().cpu().numpy() for k in keys]

        # Tolerances: ONNX can differ slightly from eager PyTorch.
        # Keep them tight enough to catch preprocessing/export regressions.
        atol = 1e-4
        rtol = 1e-4

        for key, pt, ort_val in zip(keys, pt_arrays, ort_outs):
            self.assertEqual(pt.shape, ort_val.shape, msg=f"shape mismatch for {key}: {pt.shape} vs {ort_val.shape}")
            diff = np.abs(pt - ort_val)
            max_abs = float(diff.max()) if diff.size else 0.0
            denom = np.maximum(np.abs(pt), 1e-12)
            max_rel = float((diff / denom).max()) if diff.size else 0.0
            ok = np.allclose(pt, ort_val, rtol=rtol, atol=atol)
            if not ok:
                self.fail(
                    f"ONNX parity failed for {key}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, atol={atol}, rtol={rtol}"
                )


if __name__ == "__main__":
    unittest.main()
