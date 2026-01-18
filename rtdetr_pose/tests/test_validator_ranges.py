import tempfile
import unittest
from pathlib import Path

from rtdetr_pose.validator import validate_sample


def _sample_base(image_path: Path):
    return {
        "image_path": str(image_path),
        "labels": [{"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
        "mask_path": None,
        "depth_path": None,
        "pose": None,
        "intrinsics": None,
    }


class TestValidatorRanges(unittest.TestCase):
    def test_mask_nonbinary_int_raises(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "img.jpg"
            img.write_bytes(b"")
            sample = _sample_base(img)
            sample["mask_path"] = [[0, 2], [0, 0]]
            with self.assertRaisesRegex(ValueError, "mask must be binary"):
                validate_sample(sample, strict=False, check_content=False, check_ranges=True)

    def test_mask_allows_0_255(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "img.jpg"
            img.write_bytes(b"")
            sample = _sample_base(img)
            sample["mask_path"] = [[0, 255], [0, 0]]
            validate_sample(sample, strict=False, check_content=False, check_ranges=True)

    def test_mask_float_nonbinary_raises(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "img.jpg"
            img.write_bytes(b"")
            sample = _sample_base(img)
            sample["mask_path"] = [[0.0, 0.3], [1.0, 0.0]]
            with self.assertRaisesRegex(ValueError, "mask must be binary"):
                validate_sample(sample, strict=False, check_content=False, check_ranges=True)

    def test_depth_negative_raises(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "img.jpg"
            img.write_bytes(b"")
            sample = _sample_base(img)
            sample["depth_path"] = [[0.0, -1.0], [0.0, 0.0]]
            with self.assertRaisesRegex(ValueError, "depth must be non-negative"):
                validate_sample(sample, strict=False, check_content=False, check_ranges=True)


if __name__ == "__main__":
    unittest.main()
