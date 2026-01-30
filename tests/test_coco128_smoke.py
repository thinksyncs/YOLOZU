import unittest
from pathlib import Path


class TestCoco128Smoke(unittest.TestCase):
    def setUp(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.data_dir = repo_root / "data" / "coco128"

    def test_dataset_layout(self):
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        self.assertTrue(
            (self.data_dir / "images" / "train2017").is_dir(),
            "images/train2017 missing",
        )
        self.assertTrue(
            (self.data_dir / "labels" / "train2017").is_dir(),
            "labels/train2017 missing",
        )

    def test_label_format(self):
        if not self.data_dir.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        images_dir = self.data_dir / "images" / "train2017"
        labels_dir = self.data_dir / "labels" / "train2017"
        images = sorted(images_dir.glob("*.jpg"))
        self.assertGreaterEqual(len(images), 10, "expected at least 10 images")

        for img_path in images[:5]:
            label_path = labels_dir / f"{img_path.stem}.txt"
            self.assertTrue(label_path.is_file(), f"missing label for {img_path.name}")
            contents = label_path.read_text().strip()
            if not contents:
                continue
            for line in contents.splitlines():
                parts = line.split()
                self.assertEqual(len(parts), 5, f"bad label line: {line}")
                class_id = float(parts[0])
                self.assertTrue(class_id.is_integer(), f"class id not integer: {class_id}")
                for val in parts[1:]:
                    coord = float(val)
                    self.assertGreaterEqual(coord, 0.0, f"coord < 0: {coord}")
                    self.assertLessEqual(coord, 1.0, f"coord > 1: {coord}")


if __name__ == "__main__":
    unittest.main()
