"""Learning rate scheduler factory for flexible training configurations.

Supports:
- Cosine annealing with warmup
- OneCycleLR
- MultiStepLR
- Linear warmup wrapper for any scheduler
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch.optim.lr_scheduler import _LRScheduler
except ImportError:  # pragma: no cover
    torch = None
    _LRScheduler = object


class LinearWarmupWrapper(_LRScheduler):
    """Wraps a scheduler to add linear warmup at the beginning.

    During warmup (first `warmup_steps` steps), LR increases linearly from
    `warmup_init_lr` to the base LR. After warmup, the wrapped scheduler
    takes over.
    """

    def __init__(
        self,
        optimizer: "torch.optim.Optimizer",
        warmup_steps: int,
        warmup_init_lr: float = 0.0,
        after_scheduler: "_LRScheduler | None" = None,
        last_epoch: int = -1,
    ):
        """Initialize LinearWarmupWrapper.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            warmup_init_lr: Initial LR at step 0
            after_scheduler: Scheduler to use after warmup (optional)
            last_epoch: The index of last epoch
        """
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        self.after_scheduler = after_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        # Use last_epoch which is properly maintained by the parent class
        # last_epoch starts at -1, first step() call makes it 0
        current_step = self.last_epoch + 1
        
        if current_step < self.warmup_steps:
            # Linear warmup
            if self.warmup_steps <= 0:
                return [group["lr"] for group in self.optimizer.param_groups]
            alpha = current_step / self.warmup_steps
            return [self.warmup_init_lr + (base_lr - self.warmup_init_lr) * alpha for base_lr in self.base_lrs]
        else:
            # After warmup, use the wrapped scheduler if available
            if self.after_scheduler is not None:
                return self.after_scheduler.get_last_lr()
            else:
                return self.base_lrs

    def step(self, epoch=None):
        """Step the scheduler."""
        # Step the wrapped scheduler first if we're past warmup
        # Use last_epoch to check since it hasn't been incremented yet
        if self.last_epoch >= self.warmup_steps and self.after_scheduler is not None:
            self.after_scheduler.step(epoch)
        # Then step this scheduler, which will call get_lr()
        super().step(epoch)


def build_scheduler(
    optimizer: "torch.optim.Optimizer",
    *,
    scheduler: str = "none",
    total_steps: int = 1000,
    warmup_steps: int = 0,
    warmup_init_lr: float = 0.0,
    min_lr: float = 0.0,
    milestones: list[int] | None = None,
    gamma: float = 0.1,
    **kwargs,
) -> "_LRScheduler | None":
    """Build a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler: Scheduler type ("none", "cosine", "onecycle", "multistep")
        total_steps: Total training steps
        warmup_steps: Linear warmup steps (applied to all schedulers)
        warmup_init_lr: Initial LR at step 0 during warmup
        min_lr: Minimum LR for cosine scheduler
        milestones: Step indices for MultiStepLR
        gamma: Multiplicative factor for MultiStepLR
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Scheduler instance or None if scheduler="none"
    """
    if torch is None:
        raise RuntimeError("torch is required for build_scheduler")

    scheduler = scheduler.lower()

    if scheduler == "none":
        # No scheduler, just warmup if requested
        if warmup_steps > 0:
            return LinearWarmupWrapper(optimizer, warmup_steps, warmup_init_lr)
        return None

    # Build the main scheduler
    main_scheduler = None

    if scheduler == "cosine":
        # Cosine annealing from base_lr to min_lr over total_steps
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr, **kwargs
        )

    elif scheduler == "onecycle":
        # OneCycleLR: ramps up to max_lr then down
        max_lr = kwargs.get("max_lr")
        if max_lr is None:
            # Use the base lr from optimizer as max_lr
            max_lr = [group["lr"] for group in optimizer.param_groups]

        main_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=kwargs.get("pct_start", 0.3),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
            **{k: v for k, v in kwargs.items() if k not in ["max_lr", "pct_start", "anneal_strategy"]},
        )

    elif scheduler == "multistep":
        # MultiStepLR: decay at specific milestones
        if milestones is None:
            milestones = []
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, **kwargs)

    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler}. " f"Choose 'none', 'cosine', 'onecycle', or 'multistep'."
        )

    # Wrap with linear warmup if requested
    if warmup_steps > 0:
        return LinearWarmupWrapper(optimizer, warmup_steps, warmup_init_lr, after_scheduler=main_scheduler)

    return main_scheduler


class EMA:
    """Exponential Moving Average for model weights.

    Maintains a shadow copy of model weights that are updated with EMA.
    Can be used for evaluation to potentially improve stability/performance.
    """

    def __init__(self, model: "torch.nn.Module", decay: float = 0.999):
        """Initialize EMA.

        Args:
            model: The model to track
            decay: EMA decay factor (default: 0.999)
        """
        if torch is None:
            raise RuntimeError("torch is required for EMA")

        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights with EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        """Return state dict for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]):
        """Load state from checkpoint."""
        self.decay = state.get("decay", self.decay)
        self.shadow = state.get("shadow", {})
