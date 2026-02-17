import importlib.util
import inspect
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


def _has_weights_only():
    if torch is None:
        return False
    try:
        sig = inspect.signature(torch.load)
    except Exception:  # pragma: no cover
        return False
    return "weights_only" in sig.parameters


@unittest.skipIf(torch is None or not _has_weights_only(), "torch.load(weights_only) not available")
class TestTrainMinimalDerppTeacherFormats(unittest.TestCase):
    def test_load_pt_dict(self):
        mod = _load_train_minimal_module()
        dataset = mod.ManifestDataset([], derpp_enabled=True, derpp_teacher_key="teacher", derpp_keys=("logits", "bbox"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "teacher.pt"
            teacher = {"logits": torch.randn(1, 5), "bbox": torch.randn(1, 4)}
            torch.save(teacher, path)

            loaded = dataset._load_derpp_teacher(path)
            self.assertIsInstance(loaded, dict)
            assert loaded is not None
            self.assertIn("logits", loaded)
            self.assertIn("bbox", loaded)
            self.assertEqual(tuple(loaded["logits"].shape), (5,))
            self.assertEqual(tuple(loaded["bbox"].shape), (4,))
            self.assertEqual(loaded["logits"].dtype, torch.float32)
            self.assertEqual(str(loaded["logits"].device), "cpu")

    def test_load_pth_tensor_single_key(self):
        mod = _load_train_minimal_module()
        dataset = mod.ManifestDataset([], derpp_enabled=True, derpp_teacher_key="teacher", derpp_keys=("logits",))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "teacher.pth"
            torch.save(torch.ones(1, 3), path)

            loaded = dataset._load_derpp_teacher(str(path))
            self.assertIsInstance(loaded, dict)
            assert loaded is not None
            self.assertIn("logits", loaded)
            self.assertEqual(tuple(loaded["logits"].shape), (3,))
            self.assertTrue(torch.allclose(loaded["logits"], torch.ones(3)))


if __name__ == "__main__":
    unittest.main()
