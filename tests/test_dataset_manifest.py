import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.dataset import build_manifest


class TestDatasetManifest(unittest.TestCase):
    def test_manifest_non_empty(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = build_manifest(repo_root / "data" / "coco128")
        self.assertIn("images", manifest)
        self.assertGreater(len(manifest["images"]), 10)
        first = manifest["images"][0]
        self.assertIn("image", first)
        self.assertIn("labels", first)


if __name__ == "__main__":
    unittest.main()
