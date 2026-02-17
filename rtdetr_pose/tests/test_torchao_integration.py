import sys
from pathlib import Path
import unittest


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


class TestTorchAOIntegration(unittest.TestCase):
    def _torchao_available(self) -> bool:
        try:
            import importlib

            importlib.import_module("torchao.quantization")
            return True
        except Exception:
            return False

    def test_apply_torchao_quantization_noop_when_missing(self):
        from rtdetr_pose.torchao_integration import apply_torchao_quantization

        obj = object()
        out, report = apply_torchao_quantization(obj, recipe="int8wo", required=False)
        self.assertIs(out, obj)
        self.assertFalse(bool(report.enabled))
        self.assertIn(report.reason, ("torchao_not_installed", "disabled", "config_not_found", "api_not_found"))

    def test_apply_torchao_quantization_required_raises_when_missing(self):
        if self._torchao_available():
            self.skipTest("torchao is installed; missing-dep behavior not applicable")
        from rtdetr_pose.torchao_integration import apply_torchao_quantization

        with self.assertRaises(RuntimeError):
            apply_torchao_quantization(object(), recipe="int8wo", required=True)


if __name__ == "__main__":
    unittest.main()
