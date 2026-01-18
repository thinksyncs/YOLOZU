import json
import tempfile
import unittest
from pathlib import Path

from rtdetr_pose.dataset import build_manifest
from rtdetr_pose.validator import validate_manifest


class TestValidatorMultiObject(unittest.TestCase):
    def test_multi_object_per_instance_mask_depth_pose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images" / "train2017"
            labels_dir = root / "labels" / "train2017"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            image_path = images_dir / "000010.jpg"
            image_path.write_bytes(b"")

            label_path = labels_dir / "000010.txt"
            # Two instances
            label_path.write_text(
                "0 0.5 0.5 0.2 0.2\n"
                "1 0.5 0.6 0.2 0.2\n"
            )

            meta = {
                "M": [
                    [[0, 1], [0, 1]],
                    [[0, 0], [1, 1]],
                ],
                "D_obj": [
                    [[0.0, 1.2], [0.0, 1.3]],
                    [[0.0, 0.0], [1.0, 1.1]],
                ],
                "pose": [
                    {
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "t": [0.0, 0.0, 1.0],
                    },
                    {
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "t": [0.0, 0.0, 1.0],
                    },
                ],
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0},
            }
            (labels_dir / "000010.json").write_text(json.dumps(meta))

            manifest = build_manifest(root, split="train2017")
            validate_manifest(manifest, strict=True, check_content=True, check_ranges=True)

    def test_multi_object_length_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images" / "train2017"
            labels_dir = root / "labels" / "train2017"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            image_path = images_dir / "000011.jpg"
            image_path.write_bytes(b"")

            label_path = labels_dir / "000011.txt"
            label_path.write_text(
                "0 0.5 0.5 0.2 0.2\n"
                "1 0.4 0.6 0.1 0.1\n"
            )

            meta = {
                "M": [[[0, 1], [0, 1]]],
                "D_obj": [[[0.0, 1.2], [0.0, 1.3]]],
                "pose": {
                    "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "t": [0.0, 0.0, 1.0],
                },
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0},
            }
            (labels_dir / "000011.json").write_text(json.dumps(meta))

            manifest = build_manifest(root, split="train2017")
            with self.assertRaisesRegex(ValueError, "must have length 2"):
                validate_manifest(manifest, strict=False, check_content=False, check_ranges=False)


if __name__ == "__main__":
    unittest.main()
