import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.dataset import build_manifest


class TestDatasetSegPolygons(unittest.TestCase):
    def test_parse_segment_polygon_labels(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td) / "dataset"
            images_dir = root / "images" / "train2017"
            labels_dir = root / "labels" / "train2017"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            (images_dir / "0001.jpg").write_bytes(b"")
            # class + polygon(x,y)*4 (normalized coords)
            (labels_dir / "0001.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n",
                encoding="utf-8",
            )

            manifest = build_manifest(root, split="train2017", label_format="segment")
            records = manifest.get("images") or []
            self.assertEqual(len(records), 1)
            labels = records[0].get("labels") or []
            self.assertEqual(len(labels), 1)
            lab = labels[0]
            self.assertIn("polygon", lab)
            self.assertEqual(len(lab["polygon"]), 8)
            self.assertAlmostEqual(float(lab["cx"]), 0.5, places=6)
            self.assertAlmostEqual(float(lab["cy"]), 0.5, places=6)
            self.assertAlmostEqual(float(lab["w"]), 0.8, places=6)
            self.assertAlmostEqual(float(lab["h"]), 0.8, places=6)

    def test_dataset_json_label_format_propagates(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td) / "dataset"
            images_dir = root / "images" / "train2017"
            labels_dir = root / "labels" / "train2017"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            (images_dir / "0001.jpg").write_bytes(b"")
            (labels_dir / "0001.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n",
                encoding="utf-8",
            )

            wrapper = Path(td) / "wrapper"
            wrapper.mkdir(parents=True, exist_ok=True)
            payload = {
                "images_dir": str(images_dir),
                "labels_dir": str(labels_dir),
                "split": "train2017",
                "label_format": "segment",
            }
            (wrapper / "dataset.json").write_text(json.dumps(payload), encoding="utf-8")

            manifest = build_manifest(wrapper)
            labels = (manifest.get("images") or [])[0]["labels"]
            self.assertIn("polygon", labels[0])


if __name__ == "__main__":
    unittest.main()

