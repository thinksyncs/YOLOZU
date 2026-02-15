from __future__ import annotations

from dataclasses import dataclass


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


class LoRAConv2d(nn.Module):
    """A drop-in replacement for nn.Conv2d with LoRA update (1x1 only).

    For a base 1x1 conv:
      y = conv2d(x, W, b) + scale * conv2d(conv2d(dropout(x), A), B)

    Where A is (r, in_channels, 1, 1) and B is (out_channels, r, 1, 1).
    """

    def __init__(
        self,
        base: "nn.Conv2d",
        *,
        r: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        if _IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError("torch is required for LoRA") from _IMPORT_ERROR
        super().__init__()
        if not isinstance(base, nn.Conv2d):
            raise TypeError("LoRAConv2d expects an nn.Conv2d")
        if tuple(getattr(base, "kernel_size", ())) != (1, 1):
            raise ValueError("LoRAConv2d currently supports only 1x1 conv")
        if int(getattr(base, "groups", 1)) != 1:
            raise ValueError("LoRAConv2d currently supports only groups=1")
        if int(r) <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.in_channels = int(base.in_channels)
        self.out_channels = int(base.out_channels)
        self.r = int(r)
        self.alpha = float(alpha) if alpha is not None else float(r)
        self.dropout_p = float(dropout)
        self.scale = float(self.alpha) / float(self.r)

        self.stride = tuple(base.stride)
        self.padding = tuple(base.padding)
        self.dilation = tuple(base.dilation)
        self.groups = int(base.groups)

        # Base parameters.
        self.weight = base.weight
        self.bias = base.bias

        # LoRA parameters.
        self.lora_A = nn.Parameter(torch.empty((self.r, self.in_channels, 1, 1)))
        self.lora_B = nn.Parameter(torch.empty((self.out_channels, self.r, 1, 1)))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        x_d = self.dropout(x)
        delta = F.conv2d(
            x_d,
            self.lora_A,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        delta = F.conv2d(delta, self.lora_B, bias=None, stride=1, padding=0, dilation=1, groups=1)
        return y + delta * self.scale


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
    allowed_targets = ("head", "all_linear", "all_conv1x1", "all_linear_conv1x1")
    if target not in allowed_targets:
        raise ValueError("target must be one of: head, all_linear, all_conv1x1, all_linear_conv1x1")

    replaced = 0
    linear_enabled = target in ("head", "all_linear", "all_linear_conv1x1")
    conv_enabled = target in ("all_conv1x1", "all_linear_conv1x1")
    # We need to replace modules by walking parent modules.
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            is_linear = isinstance(child, nn.Linear)
            is_conv1x1 = isinstance(child, nn.Conv2d) and tuple(getattr(child, "kernel_size", ())) == (1, 1) and int(getattr(child, "groups", 1)) == 1
            if not (is_linear or is_conv1x1):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name

            if target == "head":
                if not (full_name.startswith("head.") or full_name.startswith("offset_head.") or full_name.startswith("k_head.")):
                    continue

            if is_linear and linear_enabled:
                setattr(parent, child_name, LoRALinear(child, r=int(r), alpha=alpha, dropout=float(dropout)))
                replaced += 1
            elif is_conv1x1 and conv_enabled:
                setattr(parent, child_name, LoRAConv2d(child, r=int(r), alpha=alpha, dropout=float(dropout)))
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
        if isinstance(module, LoRAConv2d):
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
