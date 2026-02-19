from __future__ import annotations

from typing import Callable, Dict

from ...backbone_interface import BaseBackbone

_BACKBONES: Dict[str, Callable[..., BaseBackbone]] = {}


def register_backbone(name: str):
    key = str(name).strip().lower()

    def _decorator(factory: Callable[..., BaseBackbone]):
        _BACKBONES[key] = factory
        return factory

    return _decorator


def build_backbone(name: str, **kwargs) -> BaseBackbone:
    key = str(name).strip().lower()
    if key not in _BACKBONES:
        raise ValueError(f"unknown backbone: {name}. available={sorted(_BACKBONES.keys())}")
    return _BACKBONES[key](**kwargs)


def list_backbones() -> list[str]:
    return sorted(_BACKBONES.keys())
