import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.image_size import get_image_size


class TestImageSize(unittest.TestCase):
    def test_jpeg_size_from_coco128(self):
        dataset_dir = Path(__file__).resolve().parents[1] / "data" / "coco128" / "images" / "train2017"
        images = sorted(dataset_dir.glob("*.jpg"))
        self.assertTrue(images, "expected coco128 images to exist")
        w, h = get_image_size(images[0])
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)


if __name__ == "__main__":
    unittest.main()

