import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestDatasetValidator(unittest.TestCase):
    def test_strict_rejects_out_of_range_bbox(self):
        from PIL import Image

        from yolozu.dataset import build_manifest
        from yolozu.dataset_validator import validate_dataset_records

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images" / "train2017").mkdir(parents=True)
            (root / "labels" / "train2017").mkdir(parents=True)

            img = root / "images" / "train2017" / "0001.jpg"
            Image.new("RGB", (10, 10)).save(img)

            (root / "labels" / "train2017" / "0001.txt").write_text("0 1.2 0.5 0.2 0.2\n")

            manifest = build_manifest(root, split="train2017")
            res = validate_dataset_records(manifest["images"], strict=True, mode="fail")
            self.assertFalse(res.ok())
            self.assertTrue(any("out of range" in e for e in res.errors))

    def test_warn_mode_downgrades_errors(self):
        from PIL import Image

        from yolozu.dataset import build_manifest
        from yolozu.dataset_validator import validate_dataset_records

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images" / "train2017").mkdir(parents=True)
            (root / "labels" / "train2017").mkdir(parents=True)

            img = root / "images" / "train2017" / "0001.jpg"
            Image.new("RGB", (10, 10)).save(img)

            (root / "labels" / "train2017" / "0001.txt").write_text("0 1.2 0.5 0.2 0.2\n")

            manifest = build_manifest(root, split="train2017")
            res = validate_dataset_records(manifest["images"], strict=True, mode="warn")
            self.assertTrue(res.ok())
            self.assertEqual(res.errors, [])
            self.assertTrue(any("out of range" in w for w in res.warnings))


if __name__ == "__main__":
    unittest.main()
