"""Unit tests for optimizer and scheduler factory modules."""

import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@unittest.skipIf(torch is None, "torch not installed")
class TestOptimFactory(unittest.TestCase):
    """Test optimizer factory functions."""

    def setUp(self):
        """Create a simple test model."""
        if torch is not None and nn is not None:
            # Simple model with backbone and head naming convention
            self.model = nn.Sequential(
                nn.Linear(10, 20, bias=True),  # Will be treated as head
                nn.BatchNorm1d(20),            # Norm layer
                nn.ReLU(),
                nn.Linear(20, 10, bias=True),  # Will be treated as head
            )
            # Add some named modules for backbone/head detection
            self.model.add_module("backbone_layer", nn.Linear(10, 10))
            self.model.add_module("head_layer", nn.Linear(10, 10))

    def test_build_optimizer_adamw(self):
        """Test building AdamW optimizer."""
        from rtdetr_pose.rtdetr_pose.optim_factory import build_optimizer
        
        optim = build_optimizer(
            self.model,
            optimizer="adamw",
            lr=1e-4,
            weight_decay=0.01,
        )
        
        self.assertIsInstance(optim, torch.optim.AdamW)
        self.assertEqual(optim.defaults["lr"], 1e-4)
        self.assertEqual(optim.defaults["weight_decay"], 0.01)

    def test_build_optimizer_sgd(self):
        """Test building SGD optimizer with momentum and nesterov."""
        from rtdetr_pose.rtdetr_pose.optim_factory import build_optimizer
        
        optim = build_optimizer(
            self.model,
            optimizer="sgd",
            lr=0.1,
            weight_decay=1e-4,
            momentum=0.9,
            nesterov=True,
        )
        
        self.assertIsInstance(optim, torch.optim.SGD)
        self.assertEqual(optim.defaults["momentum"], 0.9)
        self.assertTrue(optim.defaults["nesterov"])

    def test_param_groups(self):
        """Test parameter groups with different lr/wd for backbone and head."""
        from rtdetr_pose.rtdetr_pose.optim_factory import build_optimizer
        
        optim = build_optimizer(
            self.model,
            optimizer="adamw",
            lr=1e-4,
            weight_decay=0.01,
            use_param_groups=True,
            backbone_lr_mult=0.1,
            head_lr_mult=1.0,
            backbone_wd_mult=0.5,
            head_wd_mult=1.0,
        )
        
        # Should have multiple param groups
        self.assertGreater(len(optim.param_groups), 1)
        
        # Check that we have groups with different lr multipliers
        # Backbone groups should have lr = base_lr * 0.1 = 1e-5
        # Head groups should have lr = base_lr * 1.0 = 1e-4
        lrs = [group["lr"] for group in optim.param_groups]
        self.assertIn(1e-5, lrs, "Should have backbone lr")
        self.assertIn(1e-4, lrs, "Should have head lr")

    def test_wd_exclude_bias(self):
        """Test that bias parameters have wd=0 when wd_exclude_bias=True."""
        from rtdetr_pose.rtdetr_pose.optim_factory import build_param_groups
        
        groups = build_param_groups(
            self.model,
            base_lr=1e-4,
            base_wd=0.01,
            wd_exclude_bias=True,
        )
        
        # Should have groups with wd=0 (for bias/norm layers)
        wds = [group["weight_decay"] for group in groups]
        self.assertIn(0.0, wds, "Should have groups with wd=0 for bias/norm")

    def test_is_norm_layer(self):
        """Test norm layer detection."""
        from rtdetr_pose.rtdetr_pose.optim_factory import is_norm_layer
        
        bn = nn.BatchNorm1d(10)
        ln = nn.LayerNorm(10)
        linear = nn.Linear(10, 10)
        
        self.assertTrue(is_norm_layer(bn))
        self.assertTrue(is_norm_layer(ln))
        self.assertFalse(is_norm_layer(linear))


@unittest.skipIf(torch is None, "torch not installed")
class TestSchedFactory(unittest.TestCase):
    """Test scheduler factory functions."""

    def setUp(self):
        """Create a simple optimizer for testing."""
        if torch is not None and nn is not None:
            model = nn.Linear(10, 10)
            self.optim = torch.optim.SGD(model.parameters(), lr=0.1)

    def test_build_scheduler_none(self):
        """Test building no scheduler."""
        from rtdetr_pose.rtdetr_pose.sched_factory import build_scheduler
        
        sched = build_scheduler(
            self.optim,
            scheduler="none",
            total_steps=100,
        )
        
        self.assertIsNone(sched)

    def test_build_scheduler_cosine(self):
        """Test building cosine annealing scheduler."""
        from rtdetr_pose.rtdetr_pose.sched_factory import build_scheduler
        
        sched = build_scheduler(
            self.optim,
            scheduler="cosine",
            total_steps=100,
            min_lr=1e-6,
        )
        
        self.assertIsNotNone(sched)

    def test_build_scheduler_onecycle(self):
        """Test building OneCycleLR scheduler."""
        from rtdetr_pose.rtdetr_pose.sched_factory import build_scheduler
        
        sched = build_scheduler(
            self.optim,
            scheduler="onecycle",
            total_steps=100,
        )
        
        self.assertIsNotNone(sched)

    def test_build_scheduler_multistep(self):
        """Test building MultiStepLR scheduler."""
        from rtdetr_pose.rtdetr_pose.sched_factory import build_scheduler
        
        sched = build_scheduler(
            self.optim,
            scheduler="multistep",
            total_steps=100,
            milestones=[30, 60, 90],
            gamma=0.1,
        )
        
        self.assertIsNotNone(sched)

    def test_warmup_wrapper(self):
        """Test linear warmup wrapper."""
        from rtdetr_pose.rtdetr_pose.sched_factory import build_scheduler
        
        sched = build_scheduler(
            self.optim,
            scheduler="cosine",
            total_steps=100,
            warmup_steps=10,
            warmup_init_lr=1e-6,
            min_lr=1e-6,
        )
        
        self.assertIsNotNone(sched)
        
        # Step scheduler and check LR changes
        initial_lr = self.optim.param_groups[0]["lr"]
        sched.step()
        after_step_lr = self.optim.param_groups[0]["lr"]
        
        # LR should change after stepping
        self.assertNotEqual(initial_lr, after_step_lr)

    def test_ema_update(self):
        """Test EMA update functionality."""
        from rtdetr_pose.rtdetr_pose.sched_factory import EMA
        
        model = nn.Linear(10, 10)
        
        # Initialize EMA
        ema = EMA(model, decay=0.9)
        
        # Store original weights
        original_weight = model.weight.data.clone()
        
        # Modify model weights
        model.weight.data += 1.0
        
        # Update EMA
        ema.update()
        
        # EMA shadow should be different from both original and current
        self.assertFalse(torch.allclose(ema.shadow["weight"], original_weight))
        self.assertFalse(torch.allclose(ema.shadow["weight"], model.weight.data))

    def test_ema_apply_restore(self):
        """Test EMA apply and restore functionality."""
        from rtdetr_pose.rtdetr_pose.sched_factory import EMA
        
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.9)
        
        # Store original weights
        original_weight = model.weight.data.clone()
        
        # Update EMA a few times
        for _ in range(5):
            model.weight.data += 0.1
            ema.update()
        
        # Apply shadow weights
        ema.apply_shadow()
        shadow_weight = model.weight.data.clone()
        
        # Restore original
        ema.restore()
        restored_weight = model.weight.data.clone()
        
        # Shadow and restored should be different
        self.assertFalse(torch.allclose(shadow_weight, restored_weight))


if __name__ == "__main__":
    unittest.main()
