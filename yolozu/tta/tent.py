from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

from .base import TTARunner
from .ttt_mim import _count_params, select_parameters


@dataclass
class TentConfig:
    lr: float = 1e-4
    include: Iterable[str] | None = None
    exclude: Iterable[str] | None = None
    update_filter: str = "all"


def _ensure_torch():
    if torch is None or F is None:
        raise RuntimeError("torch is required for TentRunner")


def _extract_logits(output: Any) -> "torch.Tensor":
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        if "pred" in output:
            return output["pred"]
    return output


def _entropy(logits: "torch.Tensor") -> "torch.Tensor":
    probs = F.softmax(logits, dim=-1)
    logp = torch.log(torch.clamp(probs, min=1e-12))
    return -(probs * logp).sum(dim=-1).mean()


class TentRunner(TTARunner):
    def __init__(self, model: "nn.Module", *, config: TentConfig | None = None):
        _ensure_torch()
        self.model = model
        self.config = config or TentConfig()
        params = select_parameters(
            model,
            update_filter=self.config.update_filter,
            include=self.config.include,
            exclude=self.config.exclude,
        )
        if not params:
            raise ValueError("no parameters selected for Tent")
        self.optimizer = torch.optim.Adam(params, lr=float(self.config.lr))
        self.updated_param_count = _count_params(params)

    def reset(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.config.lr)

    def adapt_step(self, batch: Any) -> dict[str, float]:
        _ensure_torch()
        self.model.train()
        output = self.model(batch)
        logits = _extract_logits(output)
        loss = _entropy(logits)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss_entropy": float(loss.detach().cpu())}

    def maybe_log(self) -> dict[str, Any] | None:
        return {"updated_param_count": int(self.updated_param_count)}
