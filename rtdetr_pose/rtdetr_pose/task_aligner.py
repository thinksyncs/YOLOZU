from __future__ import annotations

import math
from typing import Any


def uncertainty_weighted(loss: float, log_sigma: float) -> float:
    """Uncertainty-weighted loss term (Kendall et al.).

    Uses: exp(-s) * L + s, where s is a learned log-variance proxy.
    """

    s = float(log_sigma)
    return math.exp(-s) * float(loss) + s


def align_losses_uncertainty(
    *,
    losses: dict[str, Any],
    log_sigmas: dict[str, Any],
    mapping: dict[str, str] | None = None,
) -> dict[str, float]:
    """Align per-task losses using learned uncertainty scalars.

    Parameters:
      - losses: dict with scalar (float-like) entries such as {"loss_z": 0.1, "loss_rot": 0.2}
      - log_sigmas: dict with scalar (float-like) entries such as {"z": 0.0, "rot": -0.5}
      - mapping: optional mapping from loss key -> sigma key. Default maps:
          {"loss_z": "z", "loss_rot": "rot"}
    """

    mapping = mapping or {"loss_z": "z", "loss_rot": "rot"}
    out: dict[str, float] = {}
    for loss_key, sigma_key in mapping.items():
        if loss_key not in losses:
            continue
        if sigma_key not in log_sigmas:
            continue
        out[f"{loss_key}_aligned"] = uncertainty_weighted(float(losses[loss_key]), float(log_sigmas[sigma_key]))
    return out

