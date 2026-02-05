"""Test-time adaptation (TTA/TTT) utilities."""

from .base import TTARunner, TTAConfig, apply_tta_transform
from .tent import TentConfig, TentRunner

__all__ = ["TTARunner", "TTAConfig", "apply_tta_transform", "TentConfig", "TentRunner"]
