from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TTTPreset:
    method: str
    steps: int
    batch_size: int
    lr: float
    update_filter: str
    max_batches: int

    # Optional safety defaults (only applied when the corresponding arg is unset).
    max_grad_norm: float | None = None
    max_update_norm: float | None = None
    max_total_update_norm: float | None = None
    max_loss_ratio: float | None = None
    max_loss_increase: float | None = None


PRESETS: dict[str, TTTPreset] = {
    # Conservative defaults intended to be safe without requiring tuning.
    # They are not meant to maximize performance.
    "safe": TTTPreset(
        method="tent",
        steps=1,
        batch_size=1,
        lr=1e-4,
        update_filter="norm_only",
        max_batches=1,
        max_grad_norm=1.0,
        max_total_update_norm=1.0,
        max_loss_ratio=3.0,
    ),
    "adapter_only": TTTPreset(
        method="tent",
        steps=1,
        batch_size=1,
        lr=1e-4,
        update_filter="adapter_only",
        max_batches=1,
        max_grad_norm=5.0,
        max_total_update_norm=5.0,
        max_loss_ratio=3.0,
    ),
    "mim_safe": TTTPreset(
        method="mim",
        steps=1,
        batch_size=1,
        lr=1e-4,
        update_filter="adapter_only",
        max_batches=1,
        max_grad_norm=5.0,
        max_total_update_norm=5.0,
        max_loss_ratio=3.0,
    ),
}


def apply_ttt_preset_args(args: Any) -> None:
    """Apply a preset to an argparse Namespace in-place.

    Presets override the core algorithm knobs (method/steps/lr/filter/max_batches).
    Safety knobs are only filled in when the user did not specify them.
    """

    preset_id = getattr(args, "ttt_preset", None)
    if not preset_id:
        return
    preset = PRESETS.get(str(preset_id))
    if preset is None:
        raise ValueError(f"unknown preset: {preset_id}")

    args.ttt_method = str(preset.method)
    args.ttt_steps = int(preset.steps)
    args.ttt_batch_size = int(preset.batch_size)
    args.ttt_lr = float(preset.lr)
    args.ttt_update_filter = str(preset.update_filter)
    args.ttt_max_batches = int(preset.max_batches)

    # Safety defaults (do not clobber explicit flags).
    if getattr(args, "ttt_max_grad_norm", None) is None and preset.max_grad_norm is not None:
        args.ttt_max_grad_norm = float(preset.max_grad_norm)
    if getattr(args, "ttt_max_update_norm", None) is None and preset.max_update_norm is not None:
        args.ttt_max_update_norm = float(preset.max_update_norm)
    if getattr(args, "ttt_max_total_update_norm", None) is None and preset.max_total_update_norm is not None:
        args.ttt_max_total_update_norm = float(preset.max_total_update_norm)

    if getattr(args, "ttt_max_loss_ratio", None) is None and getattr(args, "ttt_max_loss_increase", None) is None:
        if preset.max_loss_ratio is not None:
            args.ttt_max_loss_ratio = float(preset.max_loss_ratio)
        elif preset.max_loss_increase is not None:
            args.ttt_max_loss_increase = float(preset.max_loss_increase)

