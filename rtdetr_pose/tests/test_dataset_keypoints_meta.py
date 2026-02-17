import json
import tempfile
import unittest
from pathlib import Path

from rtdetr_pose.dataset import build_manifest


class TestDatasetKeypointsMeta(unittest.TestCase):
    def test_build_manifest_reads_keypoint_schema_from_dataset_json_coco_wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images" / "train2017").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "train2017").mkdir(parents=True, exist_ok=True)

            (root / "images" / "train2017" / "0001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (root / "labels" / "train2017" / "0001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            instances = {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 64, "height": 32}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
                "categories": [
                    {
                        "id": 1,
                        "name": "person",
                        "keypoints": ["nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder"],
                        "skeleton": [[1, 2], [1, 3], [2, 4], [3, 5]],
                    }
                ],
            }
            instances_path = root / "instances_train2017.json"
            instances_path.write_text(json.dumps(instances), encoding="utf-8")

            dataset_json = {
                "format": "coco_instances",
                "instances_json": str(instances_path),
                "images_dir": str((root / "images" / "train2017").resolve()),
                "split": "train2017",
            }
            (root / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")

            manifest = build_manifest(root, split="train2017")
            self.assertIn("keypoints_meta", manifest)
            meta = manifest["keypoints_meta"]
            self.assertEqual(meta.get("num_keypoints"), 5)
            self.assertEqual(meta.get("keypoint_names"), ["nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder"])
            self.assertEqual(meta.get("skeleton"), [[1, 2], [1, 3], [2, 4], [3, 5]])


if __name__ == "__main__":
    unittest.main()
