import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _load_train_minimal_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"
    spec = importlib.util.spec_from_file_location("rtdetr_pose_tools_train_minimal", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipIf(torch is None, "torch not installed")
class TestTrainMinimalGradAccumAMP(unittest.TestCase):
    def test_gradient_accumulation_argument(self):
        """Test that gradient accumulation steps argument is parsed correctly."""
        mod = _load_train_minimal_module()
        
        # Test default value
        args = mod.parse_args([])
        self.assertEqual(args.gradient_accumulation_steps, 1)
        
        # Test custom value
        args = mod.parse_args(["--gradient-accumulation-steps", "4"])
        self.assertEqual(args.gradient_accumulation_steps, 4)
    
    def test_amp_argument(self):
        """Test that AMP argument is parsed correctly."""
        mod = _load_train_minimal_module()
        
        # Test default value (False)
        args = mod.parse_args([])
        self.assertFalse(args.use_amp)
        
        # Test with flag enabled
        args = mod.parse_args(["--use-amp"])
        self.assertTrue(args.use_amp)
    
    def test_clip_grad_norm_exists(self):
        """Test that gradient clipping argument exists (already implemented)."""
        mod = _load_train_minimal_module()
        
        # Test default value
        args = mod.parse_args([])
        self.assertEqual(args.clip_grad_norm, 0.0)
        
        # Test custom value
        args = mod.parse_args(["--clip-grad-norm", "1.0"])
        self.assertEqual(args.clip_grad_norm, 1.0)


if __name__ == "__main__":
    unittest.main()
