import sys
from pathlib import Path
import unittest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rtdetr_pose.dataset import build_manifest
from rtdetr_pose.validator import validate_manifest


class TestDataset(unittest.TestCase):
    def test_manifest_and_validation(self):
        dataset_root = repo_root / "data" / "coco128"
        if not dataset_root.exists():
            dataset_root = repo_root.parent / "data" / "coco128"
        if not dataset_root.exists():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        manifest = build_manifest(dataset_root)
        self.assertIn("images", manifest)
        self.assertGreater(len(manifest["images"]), 10)
        validate_manifest(manifest, strict=False)


if __name__ == "__main__":
    unittest.main()
