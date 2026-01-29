import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.boxes import xyxy_abs_to_cxcywh_norm


class TestBoxes(unittest.TestCase):
    def test_xyxy_to_cxcywh_norm(self):
        # Image 100x200, bbox x:[10,30], y:[20,60]
        cx, cy, w, h = xyxy_abs_to_cxcywh_norm((10.0, 20.0, 30.0, 60.0), width=100, height=200)
        self.assertAlmostEqual(cx, 0.2, places=6)  # (10+30)/2 / 100
        self.assertAlmostEqual(cy, 0.2, places=6)  # (20+60)/2 / 200
        self.assertAlmostEqual(w, 0.2, places=6)   # 20/100
        self.assertAlmostEqual(h, 0.2, places=6)   # 40/200


if __name__ == "__main__":
    unittest.main()

