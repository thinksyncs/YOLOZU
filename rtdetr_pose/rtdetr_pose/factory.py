"""Config-driven factories for the RTDETRPose scaffold.

Goal: make it easy to swap backbones/losses without touching the training loop.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    from types import SimpleNamespace

    torch = None
    nn = SimpleNamespace(Module=object)

from .config import LossConfig, ModelConfig
from .losses import Losses
from .model import CSPResNet, ConvNormAct, RTDETRPose


def _normalize_activation_name(name: str) -> str:
    value = str(name or "silu").strip().lower().replace("-", "_")
    aliases = {
        "swish": "silu",
        "hard_swish": "hardswish",
        "leaky_relu": "leakyrelu",
    }
    return aliases.get(value, value)


def _resolve_activation_pair(cfg: ModelConfig) -> tuple[str, str]:
    preset = str(getattr(cfg, "activation_preset", "default") or "default").strip().lower()
    preset_map: dict[str, tuple[str, str]] = {
        "default": ("silu", "silu"),
        "all_silu": ("silu", "silu"),
        "all_gelu": ("gelu", "gelu"),
        "all_hardswish": ("hardswish", "hardswish"),
        "all_leakyrelu": ("leakyrelu", "leakyrelu"),
        "head_leakyrelu": ("silu", "leakyrelu"),
        "mobile_hardswish": ("hardswish", "hardswish"),
    }
    if preset not in preset_map:
        raise ValueError(
            "unknown model.activation_preset: "
            f"{preset} (supported: {sorted(preset_map.keys())})"
        )

    back, head = preset_map[preset]
    back = _normalize_activation_name(str(getattr(cfg, "backbone_activation", back) or back))
    head = _normalize_activation_name(str(getattr(cfg, "head_activation", head) or head))
    valid = {"silu", "gelu", "hardswish", "leakyrelu"}
    if back not in valid:
        raise ValueError(f"unsupported model.backbone_activation: {back}")
    if head not in valid:
        raise ValueError(f"unsupported model.head_activation: {head}")
    return back, head


class TinyCNNBackbone(nn.Module):
    """Small backbone alternative for quick CPU experiments.

    Contract: forward(x) -> list[Tensor] with 3 feature maps whose channel counts
    match stage_channels.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: tuple[int, int, int],
        activation: str = "silu",
    ):
        super().__init__()
        c3, c4, c5 = stage_channels
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, activation=activation),
            ConvNormAct(stem_channels, stem_channels, kernel_size=3, padding=1, activation=activation),
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(stem_channels, c3, kernel_size=3, stride=2, padding=1, activation=activation),
            ConvNormAct(c3, c3, kernel_size=3, padding=1, activation=activation),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(c3, c4, kernel_size=3, stride=2, padding=1, activation=activation),
            ConvNormAct(c4, c4, kernel_size=3, padding=1, activation=activation),
        )
        self.stage5 = nn.Sequential(
            ConvNormAct(c4, c5, kernel_size=3, stride=2, padding=1, activation=activation),
            ConvNormAct(c5, c5, kernel_size=3, padding=1, activation=activation),
        )

    def forward(self, x):
        x = self.stem(x)
        f3 = self.stage3(x)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return [f3, f4, f5]


def build_backbone(cfg: ModelConfig) -> tuple[Any, tuple[int, int, int]]:
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("torch is required for build_backbone")

    name = str(getattr(cfg, "backbone_name", "cspresnet") or "cspresnet").lower()
    channels = tuple(int(x) for x in (getattr(cfg, "backbone_channels", None) or [64, 128, 256]))
    if len(channels) != 3:
        raise ValueError("model.backbone_channels must have length 3")
    stage_channels = (int(channels[0]), int(channels[1]), int(channels[2]))

    kwargs = getattr(cfg, "backbone_kwargs", None)
    kwargs = kwargs if isinstance(kwargs, dict) else {}
    activation_backbone, _ = _resolve_activation_pair(cfg)

    if name in ("cspresnet", "csp_resnet"):
        stage_blocks = tuple(int(x) for x in (getattr(cfg, "stage_blocks", None) or [1, 2, 2]))
        if len(stage_blocks) != 3:
            raise ValueError("model.stage_blocks must have length 3")
        use_sppf = bool(kwargs.get("use_sppf", True))
        backbone = CSPResNet(
            stem_channels=int(getattr(cfg, "stem_channels", 32)),
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            use_sppf=use_sppf,
            activation=activation_backbone,
        )
        return backbone, stage_channels

    if name in ("tiny_cnn", "tinycnn", "simple_cnn", "simplecnn"):
        backbone = TinyCNNBackbone(
            in_channels=3,
            stem_channels=int(getattr(cfg, "stem_channels", 32)),
            stage_channels=stage_channels,
            activation=activation_backbone,
        )
        return backbone, stage_channels

    raise ValueError(f"unknown backbone_name: {name}")


def build_model(cfg: ModelConfig) -> RTDETRPose:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for build_model")

    backbone, stage_channels = build_backbone(cfg)
    activation_backbone, activation_head = _resolve_activation_pair(cfg)
    return RTDETRPose(
        # Keep cfg.num_classes as the number of foreground classes and reserve the
        # last logit as "no-object" / background (RT-DETR-style).
        num_classes=int(getattr(cfg, "num_classes", 80)) + 1,
        num_keypoints=int(getattr(cfg, "num_keypoints", 0) or 0),
        enable_mim=bool(getattr(cfg, "enable_mim", False)),
        mim_geom_channels=int(getattr(cfg, "mim_geom_channels", 2) or 2),
        hidden_dim=int(getattr(cfg, "hidden_dim", 256)),
        num_queries=int(getattr(cfg, "num_queries", 300)),
        use_uncertainty=bool(getattr(cfg, "use_uncertainty", False)),
        stem_channels=int(getattr(cfg, "stem_channels", 32)),
        backbone_channels=stage_channels,
        stage_blocks=tuple(int(x) for x in (getattr(cfg, "stage_blocks", None) or [1, 2, 2])),
        num_encoder_layers=int(getattr(cfg, "num_encoder_layers", 1)),
        num_decoder_layers=int(getattr(cfg, "num_decoder_layers", 3)),
        nhead=int(getattr(cfg, "nhead", 8)),
        encoder_dim_feedforward=getattr(cfg, "encoder_dim_feedforward", None),
        decoder_dim_feedforward=getattr(cfg, "decoder_dim_feedforward", None),
        backbone=backbone,
        backbone_activation=activation_backbone,
        head_activation=activation_head,
    )


def build_losses(cfg: LossConfig) -> Losses:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for build_losses")

    name = str(getattr(cfg, "name", "default") or "default").lower()
    if name not in ("default", "losses"):
        raise ValueError(f"unknown loss config name: {name}")

    weights = getattr(cfg, "weights", None)
    if not isinstance(weights, dict):
        weights = None
    task_aligner = str(getattr(cfg, "task_aligner", "none") or "none")
    return Losses(weights=weights, task_aligner=task_aligner)
