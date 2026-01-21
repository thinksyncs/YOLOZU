import importlib.util
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.adapter import RTDETRPoseAdapter


class TestRTDETRPoseAdapter(unittest.TestCase):
    def test_requires_torch_when_used(self):
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("torch is installed; this test covers the no-torch path")

        adapter = RTDETRPoseAdapter()
        with self.assertRaises(RuntimeError) as ctx:
            adapter.predict([{"image": "does-not-exist.jpg", "labels": []}])
        self.assertIn("requires 'torch'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
