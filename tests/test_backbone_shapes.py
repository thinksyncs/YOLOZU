import math
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
from rtdetr_pose.factory import build_backbone, build_model
from rtdetr_pose.backbone_projector import BackboneProjector


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize(
    "name,expect_torchvision",
    [
        ("cspresnet", False),
        ("cspdarknet_s", False),
        ("tiny_cnn", False),
        ("resnet50", True),
        ("convnext_tiny", True),
    ],
)
def test_backbone_contract_and_projector(name, expect_torchvision):
    if expect_torchvision:
        pytest.importorskip("torchvision")

    cfg = ModelConfig(
        backbone_name=name,
        hidden_dim=128,
        backbone_channels=[64, 128, 256],
        stage_blocks=[1, 1, 1],
        backbone={"name": name, "norm": "bn", "args": {}},
        projector={"d_model": 128},
    )
    backbone, out_channels = build_backbone(cfg)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        feats = backbone(x)

    assert len(feats) == 3
    strides = [640 // int(feat.shape[-1]) for feat in feats]
    assert strides == [8, 16, 32]
    for feat in feats:
        assert torch.isfinite(feat).all()

    proj = BackboneProjector(out_channels, d_model=128)
    with torch.no_grad():
        p_feats = proj(feats)
    assert [int(f.shape[1]) for f in p_feats] == [128, 128, 128]
    for feat in p_feats:
        assert torch.isfinite(feat).all()


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_model_build_with_nested_backbone_config():
    cfg = ModelConfig(
        num_classes=3,
        num_queries=20,
        hidden_dim=64,
        backbone={"name": "cspdarknet_s", "norm": "bn", "args": {"width_mult": 0.5, "depth_mult": 0.34}},
        projector={"d_model": 64},
        num_decoder_layers=1,
        nhead=4,
    )
    model = build_model(cfg)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out["logits"].shape[0] == 1
    assert torch.isfinite(out["bbox"]).all()
