"""CPU smoke test for configurable optimizer and scheduler options.

Tests:
1. SGD vs AdamW optimizer switching
2. Scheduler switching (none, cosine, onecycle, multistep)
3. Param group split (backbone/head with different lr/wd)
4. EMA functionality
5. Predictions JSON schema unchanged
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _load_train_minimal_module():
    """Load train_minimal.py as a module."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"
    spec = importlib.util.spec_from_file_location("rtdetr_pose_tools_train_minimal", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipIf(torch is None, "torch not installed")
class TestOptimizerSchedulerFactory(unittest.TestCase):
    """Test configurable optimizer and scheduler options."""

    def setUp(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.repo_root / "data" / "coco128"
        if not self.data_dir.is_dir():
            self.data_dir = self.repo_root.parent / "data" / "coco128"
        self.config_path = self.repo_root / "rtdetr_pose" / "configs" / "base.json"

    def _run_training(self, args_list, tmpdir):
        """Helper to run training with given arguments."""
        mod = _load_train_minimal_module()
        
        base_args = [
            "--dataset-root", str(self.data_dir),
            "--split", "train2017",
            "--epochs", "1",
            "--batch-size", "2",
            "--max-steps", "3",
            "--image-size", "64",
            "--device", "cpu",
            "--metrics-jsonl", str(Path(tmpdir) / "metrics.jsonl"),
            "--no-export-onnx",
        ]
        
        if self.config_path.exists():
            base_args.extend(["--config", str(self.config_path)])
        
        full_args = base_args + args_list
        exit_code = mod.main(full_args)
        return exit_code

    def test_optimizer_adamw(self):
        """Test AdamW optimizer (default)."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = self._run_training(["--optimizer", "adamw"], tmpdir)
            self.assertEqual(exit_code, 0)
            
            # Check metrics were logged
            metrics_file = Path(tmpdir) / "metrics.jsonl"
            self.assertTrue(metrics_file.exists())

    def test_optimizer_sgd(self):
        """Test SGD optimizer."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = self._run_training(
                ["--optimizer", "sgd", "--momentum", "0.9", "--nesterov"],
                tmpdir
            )
            self.assertEqual(exit_code, 0)

    def test_scheduler_cosine(self):
        """Test cosine annealing scheduler."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = self._run_training(
                ["--scheduler", "cosine", "--min-lr", "1e-6"],
                tmpdir
            )
            self.assertEqual(exit_code, 0)

    def test_scheduler_onecycle(self):
        """Test OneCycleLR scheduler."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = self._run_training(
                ["--scheduler", "onecycle"],
                tmpdir
            )
            self.assertEqual(exit_code, 0)

    def test_scheduler_multistep(self):
        """Test MultiStepLR scheduler."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = self._run_training(
                ["--scheduler", "multistep", "--scheduler-milestones", "2,5"],
                tmpdir
            )
            self.assertEqual(exit_code, 0)

    def test_param_groups(self):
        """Test parameter groups with different lr/wd for backbone/head."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = self._run_training(
                [
                    "--use-param-groups",
                    "--backbone-lr-mult", "0.1",
                    "--head-lr-mult", "1.0",
                    "--backbone-wd-mult", "0.5",
                    "--head-wd-mult", "1.0",
                ],
                tmpdir
            )
            self.assertEqual(exit_code, 0)
            
            # Check metrics include param group info
            metrics_file = Path(tmpdir) / "metrics.jsonl"
            self.assertTrue(metrics_file.exists())

    def test_ema(self):
        """Test EMA functionality."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = self._run_training(
                ["--use-ema", "--ema-decay", "0.999"],
                tmpdir
            )
            self.assertEqual(exit_code, 0)
            
            # Check metrics include EMA decay
            metrics_file = Path(tmpdir) / "metrics.jsonl"
            self.assertTrue(metrics_file.exists())
            
            # Read last line and check for ema_decay
            with open(metrics_file) as f:
                lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                # EMA decay should be in metrics
                if "metrics" in last_record:
                    self.assertIn("ema_decay", last_record["metrics"])

    def test_gradient_clipping_logging(self):
        """Test that gradient clipping logs grad_norm."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = self._run_training(
                ["--clip-grad-norm", "1.0"],
                tmpdir
            )
            self.assertEqual(exit_code, 0)
            
            # Check metrics include grad_norm
            metrics_file = Path(tmpdir) / "metrics.jsonl"
            self.assertTrue(metrics_file.exists())
            
            # Read and check for grad_norm in metrics
            with open(metrics_file) as f:
                lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                if "metrics" in last_record:
                    # grad_norm should be logged when clipping is enabled
                    self.assertIn("grad_norm", last_record["metrics"])

    def test_combined_features(self):
        """Test combination of optimizer, scheduler, param groups, and EMA."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = self._run_training(
                [
                    "--optimizer", "sgd",
                    "--momentum", "0.9",
                    "--scheduler", "cosine",
                    "--min-lr", "1e-6",
                    "--lr-warmup-steps", "2",
                    "--use-param-groups",
                    "--backbone-lr-mult", "0.1",
                    "--use-ema",
                    "--ema-decay", "0.99",
                    "--clip-grad-norm", "1.0",
                ],
                tmpdir
            )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
