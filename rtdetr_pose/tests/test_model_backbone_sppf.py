import unittest
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "rtdetr_pose"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from rtdetr_pose.model import RTDETRPose


@unittest.skipIf(torch is None, "torch not installed")
class TestModelBackboneSPPF(unittest.TestCase):
    def test_use_sppf_flag(self):
        model = RTDETRPose(use_sppf=False)
        self.assertIsNone(model.backbone.sppf)

        model_sppf = RTDETRPose(use_sppf=True)
        self.assertIsNotNone(model_sppf.backbone.sppf)


if __name__ == "__main__":
    unittest.main()
