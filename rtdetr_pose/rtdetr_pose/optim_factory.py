"""Optimizer factory for flexible training configurations.

Supports:
- SGD with momentum and nesterov options
- AdamW
- Param groups for backbone/head with different lr, wd
- Weight decay exclusion for bias and norm layers
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def is_norm_layer(module: "nn.Module") -> bool:
    """Check if a module is a normalization layer."""
    if nn is None:
        return False
    return isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    )


def build_param_groups(
    model: "nn.Module",
    *,
    base_lr: float = 1e-4,
    base_wd: float = 0.01,
    backbone_lr_mult: float = 1.0,
    head_lr_mult: float = 1.0,
    backbone_wd_mult: float = 1.0,
    head_wd_mult: float = 1.0,
    wd_exclude_bias: bool = True,
    wd_exclude_norm: bool = True,
) -> list[dict[str, Any]]:
    """Build parameter groups with different lr and wd for backbone and head.

    Args:
        model: The model to build param groups for
        base_lr: Base learning rate
        base_wd: Base weight decay
        backbone_lr_mult: Multiplier for backbone lr (default: 1.0)
        head_lr_mult: Multiplier for head lr (default: 1.0)
        backbone_wd_mult: Multiplier for backbone wd (default: 1.0)
        head_wd_mult: Multiplier for head wd (default: 1.0)
        wd_exclude_bias: If True, set wd=0 for bias parameters
        wd_exclude_norm: If True, set wd=0 for norm layer parameters

    Returns:
        List of parameter group dicts for optimizer
    """
    if torch is None or nn is None:
        raise RuntimeError("torch is required for build_param_groups")

    # Separate backbone and head parameters
    backbone_params_wd = []
    backbone_params_no_wd = []
    head_params_wd = []
    head_params_no_wd = []

    for name, module in model.named_modules():
        # Determine if this is a backbone or head module
        # Convention: backbone modules typically have "backbone" or "encoder" in name
        # Head modules typically have "head", "decoder", or "query" in name
        is_backbone = any(x in name.lower() for x in ["backbone", "encoder"])
        is_head = any(x in name.lower() for x in ["head", "decoder", "query", "class", "bbox"])

        # If neither backbone nor head markers found, default to head
        if not is_backbone and not is_head:
            is_head = True

        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            # Check if this parameter should have wd=0
            no_wd = False
            full_name = f"{name}.{param_name}" if name else param_name

            if wd_exclude_bias and "bias" in param_name.lower():
                no_wd = True
            if wd_exclude_norm and is_norm_layer(module):
                no_wd = True

            # Add to appropriate group
            if is_backbone:
                if no_wd:
                    backbone_params_no_wd.append(param)
                else:
                    backbone_params_wd.append(param)
            else:  # is_head
                if no_wd:
                    head_params_no_wd.append(param)
                else:
                    head_params_wd.append(param)

    # Build parameter groups
    groups = []

    if backbone_params_wd:
        groups.append(
            {
                "params": backbone_params_wd,
                "lr": base_lr * backbone_lr_mult,
                "weight_decay": base_wd * backbone_wd_mult,
                "name": "backbone_wd",
            }
        )

    if backbone_params_no_wd:
        groups.append(
            {
                "params": backbone_params_no_wd,
                "lr": base_lr * backbone_lr_mult,
                "weight_decay": 0.0,
                "name": "backbone_no_wd",
            }
        )

    if head_params_wd:
        groups.append(
            {
                "params": head_params_wd,
                "lr": base_lr * head_lr_mult,
                "weight_decay": base_wd * head_wd_mult,
                "name": "head_wd",
            }
        )

    if head_params_no_wd:
        groups.append(
            {
                "params": head_params_no_wd,
                "lr": base_lr * head_lr_mult,
                "weight_decay": 0.0,
                "name": "head_no_wd",
            }
        )

    # Fallback: if no groups created, use all parameters with base settings
    if not groups:
        groups.append(
            {
                "params": model.parameters(),
                "lr": base_lr,
                "weight_decay": base_wd,
                "name": "default",
            }
        )

    return groups


def build_optimizer(
    model: "nn.Module",
    *,
    optimizer: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.9,
    nesterov: bool = False,
    use_param_groups: bool = False,
    backbone_lr_mult: float = 1.0,
    head_lr_mult: float = 1.0,
    backbone_wd_mult: float = 1.0,
    head_wd_mult: float = 1.0,
    wd_exclude_bias: bool = True,
    wd_exclude_norm: bool = True,
    **kwargs,
) -> "torch.optim.Optimizer":
    """Build an optimizer with optional parameter groups.

    Args:
        model: The model to optimize
        optimizer: Optimizer type ("adamw" or "sgd")
        lr: Base learning rate
        weight_decay: Base weight decay
        momentum: SGD momentum (default: 0.9)
        nesterov: Use Nesterov momentum for SGD (default: False)
        use_param_groups: If True, create separate groups for backbone/head
        backbone_lr_mult: LR multiplier for backbone parameters
        head_lr_mult: LR multiplier for head parameters
        backbone_wd_mult: WD multiplier for backbone parameters
        head_wd_mult: WD multiplier for head parameters
        wd_exclude_bias: Exclude bias from weight decay
        wd_exclude_norm: Exclude norm layers from weight decay
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured optimizer instance
    """
    if torch is None:
        raise RuntimeError("torch is required for build_optimizer")

    optimizer = optimizer.lower()

    # Build parameter groups if requested
    if use_param_groups:
        param_groups = build_param_groups(
            model,
            base_lr=lr,
            base_wd=weight_decay,
            backbone_lr_mult=backbone_lr_mult,
            head_lr_mult=head_lr_mult,
            backbone_wd_mult=backbone_wd_mult,
            head_wd_mult=head_wd_mult,
            wd_exclude_bias=wd_exclude_bias,
            wd_exclude_norm=wd_exclude_norm,
        )
    else:
        # Simple single group
        param_groups = [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay, "name": "default"}]

    # Create optimizer
    if optimizer == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,  # base lr, overridden by groups
            momentum=momentum,
            weight_decay=weight_decay,  # base wd, overridden by groups
            nesterov=nesterov,
            **kwargs,
        )
    elif optimizer == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=lr,  # base lr, overridden by groups
            weight_decay=weight_decay,  # base wd, overridden by groups
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}. Choose 'sgd' or 'adamw'.")
