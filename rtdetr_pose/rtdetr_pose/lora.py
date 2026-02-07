from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None

LoRATarget = Literal["head", "all_linear", "all_conv1x1", "all_linear_conv1x1"]


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("torch is required for rtdetr_pose.lora")


@dataclass(frozen=True)
class TrainableInfo:
    lora_params: int
    bias_params: int
    total_trainable_params: int

    def to_dict(self) -> dict[str, int]:
        return {
            "lora_params": int(self.lora_params),
            "bias_params": int(self.bias_params),
            "total_trainable_params": int(self.total_trainable_params),
        }


class LoRALinear(nn.Module):
    def __init__(self, base: "nn.Linear", *, r: int, alpha: float | None, dropout: float):
        _require_torch()
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")
        if r <= 0:
            raise ValueError("r must be > 0")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha) if alpha is not None else float(r)
        self.scaling = float(self.alpha / float(self.r))

        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, self.r, bias=False)
        self.lora_up = nn.Linear(self.r, base.out_features, bias=False)

        # Initialize so that the LoRA path starts as a no-op.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(self.dropout(x))) * float(self.scaling)
        return base_out + lora_out


class LoRAConv2d1x1(nn.Module):
    def __init__(self, base: "nn.Conv2d", *, r: int, alpha: float | None, dropout: float):
        _require_torch()
        super().__init__()
        if not isinstance(base, nn.Conv2d):
            raise TypeError("base must be nn.Conv2d")
        if tuple(base.kernel_size) != (1, 1):
            raise ValueError("base must be a 1x1 conv")
        if r <= 0:
            raise ValueError("r must be > 0")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha) if alpha is not None else float(r)
        self.scaling = float(self.alpha / float(self.r))

        self.dropout = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.lora_down = nn.Conv2d(base.in_channels, self.r, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(self.r, base.out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(self.dropout(x))) * float(self.scaling)
        return base_out + lora_out


def _iter_named_children(module: "nn.Module") -> list[tuple["nn.Module", str, "nn.Module"]]:
    out: list[tuple[nn.Module, str, nn.Module]] = []
    for name, child in module.named_children():
        out.append((module, name, child))
        out.extend(_iter_named_children(child))
    return out


def apply_lora(
    model: "nn.Module",
    *,
    r: int,
    alpha: float | None = None,
    dropout: float = 0.0,
    target: LoRATarget = "head",
) -> int:
    """Replace selected layers with LoRA-wrapped variants.

    Returns the number of modules replaced.
    """

    _require_torch()
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if int(r) <= 0:
        raise ValueError("r must be > 0")
    if float(dropout) < 0.0:
        raise ValueError("dropout must be >= 0")

    wanted = str(target)
    if wanted not in ("head", "all_linear", "all_conv1x1", "all_linear_conv1x1"):
        raise ValueError("target must be one of: head, all_linear, all_conv1x1, all_linear_conv1x1")

    def is_selected(full_name: str, child: nn.Module) -> bool:
        if isinstance(child, (LoRALinear, LoRAConv2d1x1)):
            return False
        name_ok = True
        if wanted == "head":
            name_ok = "head" in full_name

        type_ok = False
        if wanted in ("head", "all_linear", "all_linear_conv1x1") and isinstance(child, nn.Linear):
            type_ok = True
        if wanted in ("head", "all_conv1x1", "all_linear_conv1x1") and isinstance(child, nn.Conv2d):
            type_ok = tuple(child.kernel_size) == (1, 1)

        return bool(name_ok and type_ok)

    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for parent, name, child in _iter_named_children(model):
        full_name = name
        try:
            # Compute a stable dotted name by searching from the root.
            for n, m in model.named_modules():
                if m is parent:
                    full_name = f"{n}.{name}" if n else name
                    break
        except Exception:
            full_name = name

        if not is_selected(full_name, child):
            continue

        if isinstance(child, nn.Linear):
            replacements.append((parent, name, LoRALinear(child, r=int(r), alpha=alpha, dropout=float(dropout))))
        elif isinstance(child, nn.Conv2d) and tuple(child.kernel_size) == (1, 1):
            replacements.append((parent, name, LoRAConv2d1x1(child, r=int(r), alpha=alpha, dropout=float(dropout))))

    for parent, name, new in replacements:
        setattr(parent, name, new)

    return int(len(replacements))


def count_trainable_params(model: "nn.Module") -> int:
    _require_torch()
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    return int(sum(int(p.numel()) for p in model.parameters() if bool(p.requires_grad)))


def mark_only_lora_as_trainable(model: "nn.Module", *, train_bias: Literal["none", "all"] = "none") -> dict[str, Any]:
    """Freeze all params except LoRA (and optionally biases)."""

    _require_torch()
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if str(train_bias) not in ("none", "all"):
        raise ValueError("train_bias must be one of: none, all")

    for p in model.parameters():
        p.requires_grad = False

    lora_params = 0
    bias_params = 0
    for name, p in model.named_parameters():
        is_lora = ".lora_down." in name or ".lora_up." in name
        is_bias = name.endswith(".bias")
        if is_lora:
            p.requires_grad = True
            lora_params += int(p.numel())
        elif train_bias == "all" and is_bias:
            p.requires_grad = True
            bias_params += int(p.numel())

    info = TrainableInfo(
        lora_params=int(lora_params),
        bias_params=int(bias_params),
        total_trainable_params=count_trainable_params(model),
    )
    return info.to_dict()
