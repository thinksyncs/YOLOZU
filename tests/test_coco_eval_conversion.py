import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.coco_eval import build_coco_ground_truth, predictions_to_coco_detections
from yolozu.dataset import build_manifest


class TestCocoEvalConversion(unittest.TestCase):
    def test_build_ground_truth_schema(self):
        dataset_root = Path(__file__).resolve().parents[1] / "data" / "coco128"
        manifest = build_manifest(dataset_root)
        records = manifest["images"][:3]

        gt, index = build_coco_ground_truth(records)
        self.assertIn("images", gt)
        self.assertIn("annotations", gt)
        self.assertIn("categories", gt)
        self.assertEqual(len(gt["images"]), len(records))

        self.assertTrue(index.image_key_to_id)
        self.assertTrue(index.class_id_to_category_id)

    def test_predictions_to_detections_mapping(self):
        dataset_root = Path(__file__).resolve().parents[1] / "data" / "coco128"
        manifest = build_manifest(dataset_root)
        records = manifest["images"][:2]

        gt, index = build_coco_ground_truth(records)
        image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt["images"]}

        # One dummy detection per image, in normalized cxcywh.
        preds = []
        for rec in records:
            preds.append(
                {
                    "image": Path(rec["image"]).name,
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        }
                    ],
                }
            )

        dt = predictions_to_coco_detections(preds, coco_index=index, image_sizes=image_sizes, bbox_format="cxcywh_norm")
        self.assertEqual(len(dt), len(records))
        for det in dt:
            self.assertIn("image_id", det)
            self.assertIn("category_id", det)
            self.assertIn("bbox", det)
            self.assertIn("score", det)


if __name__ == "__main__":
    unittest.main()

