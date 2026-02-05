from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TTTConfig:
    enabled: bool = False

    # "tent" (entropy minimization) or "mim" (masked image modeling)
    method: str = "tent"

    # Total adaptation steps to run (steps over batches).
    steps: int = 1

    # Adapter-provided loader batch size.
    batch_size: int = 1

    # Optimizer learning rate (Tent or MIM).
    lr: float = 1e-4

    # Parameter selection strategy (see yolozu/tta/ttt_mim.py: select_parameters).
    update_filter: str = "all"  # all | norm_only | adapter_only
    include: Iterable[str] | None = None
    exclude: Iterable[str] | None = None

    # Cap number of distinct batches used during adaptation.
    max_batches: int = 1

    # Seed for stochastic parts (e.g., MIM masks).
    seed: int | None = None

    # Optional JSON log path (typically used by CLI wrappers).
    log_out: str | None = None

    # MIM-specific options.
    mim_mask_prob: float = 0.6
    mim_patch_size: int = 16
    mim_mask_value: float = 0.0
