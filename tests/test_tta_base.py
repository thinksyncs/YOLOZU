import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.tta import TTARunner, TTAConfig, apply_tta_transform


class TestTTABase(unittest.TestCase):
    def test_apply_tta_transform_disabled(self):
        entries = [{"image": "x.jpg", "detections": [{"bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.1}}]}]
        cfg = TTAConfig(enabled=False, seed=1)
        out = apply_tta_transform(entries, config=cfg)
        self.assertEqual(out.entries[0]["detections"][0]["bbox"]["cx"], 0.2)
        self.assertFalse(out.warnings)

    def test_apply_tta_transform_enabled(self):
        entries = [{"image": "x.jpg", "detections": [{"bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.1}}]}]
        cfg = TTAConfig(enabled=True, seed=0, flip_prob=1.0)
        out = apply_tta_transform(entries, config=cfg)
        self.assertAlmostEqual(out.entries[0]["detections"][0]["bbox"]["cx"], 0.8)

    def test_tta_runner_interface(self):
        class Dummy(TTARunner):
            def __init__(self):
                self.steps = 0

            def adapt_step(self, batch):
                self.steps += 1
                return {"loss": 0.0}

        runner = Dummy()
        runner.reset()
        out = runner.adapt_step({"x": 1})
        self.assertIn("loss", out)
        self.assertIsNone(runner.maybe_log())


if __name__ == "__main__":
    unittest.main()
