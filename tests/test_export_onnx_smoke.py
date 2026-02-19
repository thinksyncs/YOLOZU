from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rtdetr_pose"))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from rtdetr_pose.config import ModelConfig
from rtdetr_pose.factory import build_model
from rtdetr_pose.export import export_onnx


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_export_onnx_smoke(tmp_path: Path):
    pytest.importorskip("onnx")

    model = build_model(
        ModelConfig(
            num_classes=2,
            num_queries=20,
            hidden_dim=64,
            num_decoder_layers=1,
            nhead=4,
            backbone={"name": "tiny_cnn", "norm": "bn", "args": {"stage_channels": [32, 64, 128]}},
            projector={"d_model": 64},
        )
    )
    out_path = tmp_path / "model.onnx"
    dummy = torch.zeros(1, 3, 128, 128)
    export_onnx(model, dummy, str(out_path), opset_version=17, dynamic_hw=False)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
