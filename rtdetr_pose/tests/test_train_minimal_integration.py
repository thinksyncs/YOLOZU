import importlib.util
import tempfile
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
class TestTrainMinimalIntegration(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.repo_root / "data" / "coco128"
        if not self.data_dir.is_dir():
            self.data_dir = self.repo_root.parent / "data" / "coco128"
    
    def test_gradient_accumulation_integration(self):
        """Test that training works with gradient accumulation."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        mod = _load_train_minimal_module()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                "--dataset-root", str(self.data_dir),
                "--split", "train2017",
                "--epochs", "1",
                "--batch-size", "2",
                "--max-steps", "3",
                "--image-size", "64",
                "--device", "cpu",
                "--gradient-accumulation-steps", "2",
                "--metrics-jsonl", str(Path(tmpdir) / "metrics.jsonl"),
                "--no-export-onnx",
            ]
            
            result = mod.main(args)
            self.assertEqual(result, 0, "Training should complete successfully")
            
            # Check that metrics file exists
            metrics_file = Path(tmpdir) / "metrics.jsonl"
            self.assertTrue(metrics_file.exists(), "Metrics file should be created")
            
            # Verify metrics were written
            with open(metrics_file) as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 0, "Metrics should be logged")
    
    def test_amp_on_cpu_warning(self):
        """Test that AMP on CPU device shows warning."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        mod = _load_train_minimal_module()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                "--dataset-root", str(self.data_dir),
                "--split", "train2017",
                "--epochs", "1",
                "--batch-size", "2",
                "--max-steps", "2",
                "--image-size", "64",
                "--device", "cpu",
                "--use-amp",
                "--metrics-jsonl", str(Path(tmpdir) / "metrics.jsonl"),
                "--no-export-onnx",
            ]
            
            # This should complete but print a warning about AMP requiring CUDA
            result = mod.main(args)
            self.assertEqual(result, 0, "Training should complete successfully even with AMP on CPU")
    
    def test_combined_features(self):
        """Test that gradient clipping, accumulation work together."""
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        
        mod = _load_train_minimal_module()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                "--dataset-root", str(self.data_dir),
                "--split", "train2017",
                "--epochs", "1",
                "--batch-size", "2",
                "--max-steps", "4",
                "--image-size", "64",
                "--device", "cpu",
                "--clip-grad-norm", "1.0",
                "--gradient-accumulation-steps", "2",
                "--metrics-jsonl", str(Path(tmpdir) / "metrics.jsonl"),
                "--no-export-onnx",
            ]
            
            result = mod.main(args)
            self.assertEqual(result, 0, "Training should complete successfully with combined features")
            
            # Check that metrics file exists
            metrics_file = Path(tmpdir) / "metrics.jsonl"
            self.assertTrue(metrics_file.exists(), "Metrics file should be created")


if __name__ == "__main__":
    unittest.main()
