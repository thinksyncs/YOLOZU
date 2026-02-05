from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..predictions_transform import TransformResult, apply_tta


@dataclass(frozen=True)
class TTAConfig:
    enabled: bool = False
    seed: int | None = None
    flip_prob: float = 0.5
    norm_only: bool = False


def apply_tta_transform(
    entries: Iterable[dict[str, Any]], *, config: TTAConfig
) -> TransformResult:
    return apply_tta(
        entries,
        enabled=bool(config.enabled),
        seed=config.seed,
        flip_prob=float(config.flip_prob),
        norm_only=bool(config.norm_only),
    )
