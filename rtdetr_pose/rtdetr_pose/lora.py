from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 0
    alpha: float | None = None
    dropout: float = 0.0
    target: str = "head"  # "head" or "all_linear"
    freeze_base: bool = True
    train_bias: str = "none"  # "none" or "all"


class LoRALinear(nn.Module):
    """A drop-in replacement for nn.Linear with LoRA update.

    Computes:
      y = linear(x, W, b) + scale * ( (dropout(x) @ A^T) @ B^T )

    Where A is (r, in_features) and B is (out_features, r).
    B is initialized to zeros so the initial delta is zero.
    """

    def __init__(
        self,
        base: "nn.Linear",
        *,
        r: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        if _IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError("torch is required for LoRA") from _IMPORT_ERROR
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear")
        if int(r) <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.r = int(r)
        self.alpha = float(alpha) if alpha is not None else float(r)
        self.dropout_p = float(dropout)
        self.scale = float(self.alpha) / float(self.r)

        # Base parameters (kept for forward; optionally frozen by helper).
        self.weight = base.weight
        self.bias = base.bias

        # LoRA parameters.
        self.lora_A = nn.Parameter(torch.empty((self.r, self.in_features)))
        self.lora_B = nn.Parameter(torch.empty((self.out_features, self.r)))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = F.linear(x, self.weight, self.bias)
        x_d = self.dropout(x)
        delta = F.linear(F.linear(x_d, self.lora_A, bias=None), self.lora_B, bias=None)
        return y + delta * self.scale


def _iter_named_linear_modules(model: "nn.Module") -> Iterable[tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def apply_lora(
    model: "nn.Module",
    *,
    r: int,
    alpha: float | None = None,
    dropout: float = 0.0,
    target: str = "head",
) -> int:
    """Replace selected nn.Linear layers with LoRALinear.

    target:
      - "head": only linears under model.head (and optional offset/k heads)
      - "all_linear": all nn.Linear in the module tree

    Returns number of replaced layers.
    """
    if _IMPORT_ERROR is not None:  # pragma: no cover
        raise RuntimeError("torch is required for LoRA") from _IMPORT_ERROR
    if int(r) <= 0:
        return 0
    target = str(target)
    if target not in ("head", "all_linear"):
        raise ValueError("target must be 'head' or 'all_linear'")

    replaced = 0
    # We need to replace modules by walking parent modules.
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name

            if target == "head":
                allowed_prefixes = ("head.", "offset_head.", "k_head.")
                if not (full_name == "head" or full_name.startswith(allowed_prefixes)):
                    continue
                if not (
                    full_name.startswith("head.")
                    or full_name.startswith("offset_head.")
                    or full_name.startswith("k_head.")
                ):
                    continue

            setattr(parent, child_name, LoRALinear(child, r=int(r), alpha=alpha, dropout=float(dropout)))
            replaced += 1

    return replaced


def mark_only_lora_as_trainable(model: "nn.Module", *, train_bias: str = "none") -> dict[str, int]:
    """Freeze all parameters except LoRA params (and optionally biases)."""
    if _IMPORT_ERROR is not None:  # pragma: no cover
        raise RuntimeError("torch is required for LoRA") from _IMPORT_ERROR
    train_bias = str(train_bias)
    if train_bias not in ("none", "all"):
        raise ValueError("train_bias must be 'none' or 'all'")

    for p in model.parameters():
        p.requires_grad = False

    trainable = 0
    lora_params = 0
    bias_params = 0

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            lora_params += module.lora_A.numel() + module.lora_B.numel()
            trainable += module.lora_A.numel() + module.lora_B.numel()
            if train_bias == "all" and module.bias is not None:
                module.bias.requires_grad = True
                bias_params += module.bias.numel()
                trainable += module.bias.numel()

    return {"trainable_params": int(trainable), "lora_params": int(lora_params), "bias_params": int(bias_params)}


def count_trainable_params(model: "nn.Module") -> int:
    if _IMPORT_ERROR is not None:  # pragma: no cover
        raise RuntimeError("torch is required") from _IMPORT_ERROR
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += int(p.numel())
    return int(total)
