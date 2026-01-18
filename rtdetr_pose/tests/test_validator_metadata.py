import json
import sys
import tempfile
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rtdetr_pose.dataset import build_manifest
from rtdetr_pose.validator import validate_manifest


class TestValidatorMetadata(unittest.TestCase):
    def test_metadata_sidecar_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images" / "train2017"
            labels_dir = root / "labels" / "train2017"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            image_path = images_dir / "000001.jpg"
            image_path.write_bytes(b"")
            label_path = labels_dir / "000001.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.3\n")

            mask_path = root / "masks" / "train2017" / "000001.png"
            depth_path = root / "depth" / "train2017" / "000001.npy"
            mask_path.parent.mkdir(parents=True)
            depth_path.parent.mkdir(parents=True)
            mask_path.write_bytes(b"")
            depth_path.write_bytes(b"")

            meta = {
                "mask_path": str(mask_path.relative_to(root)),
                "depth_path": str(depth_path.relative_to(root)),
                "pose": {
                    "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "t": [0.0, 0.0, 1.0],
                },
                "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
            }
            meta_path = labels_dir / "000001.json"
            meta_path.write_text(json.dumps(meta))

            manifest = build_manifest(root, split="train2017")
            self.assertEqual(len(manifest["images"]), 1)
            sample = manifest["images"][0]
            self.assertIsNotNone(sample["pose"])
            self.assertIsNotNone(sample["intrinsics"])
            self.assertIsNotNone(sample["mask_path"])
            self.assertIsNotNone(sample["depth_path"])
            validate_manifest(manifest, strict=True)


if __name__ == "__main__":
    unittest.main()
