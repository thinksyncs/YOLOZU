import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.dataset import build_manifest


class TestDatasetManifest(unittest.TestCase):
    def test_manifest_non_empty(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset_root = repo_root / "data" / "coco128"
        if not dataset_root.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        manifest = build_manifest(dataset_root)
        self.assertIn("images", manifest)
        self.assertGreater(len(manifest["images"]), 10)
        first = manifest["images"][0]
        self.assertIn("image", first)
        self.assertIn("labels", first)


if __name__ == "__main__":
    unittest.main()
