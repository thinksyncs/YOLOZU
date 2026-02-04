import unittest

from yolozu.distillation import distill_predictions
from yolozu.simple_map import evaluate_map


class TestDistillation(unittest.TestCase):
    def test_distill_adds_teacher(self):
        records = [
            {
                "image": "img1.jpg",
                "labels": [
                    {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                ],
            },
            {
                "image": "img2.jpg",
                "labels": [
                    {"class_id": 0, "cx": 0.3, "cy": 0.3, "w": 0.2, "h": 0.2},
                ],
            },
        ]

        student = [
            {
                "image": "img1.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.4, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                ],
            },
            {"image": "img2.jpg", "detections": []},
        ]

        teacher = [
            {
                "image": "img1.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                ],
            },
            {
                "image": "img2.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.9, "bbox": {"cx": 0.3, "cy": 0.3, "w": 0.2, "h": 0.2}},
                ],
            },
        ]

        base_map = evaluate_map(records, student, iou_thresholds=[0.5]).map50
        distilled, stats = distill_predictions(student, teacher, add_missing=True)
        distilled_map = evaluate_map(records, distilled, iou_thresholds=[0.5]).map50

        self.assertGreaterEqual(distilled_map, base_map)
        self.assertGreater(stats.added, 0)


if __name__ == "__main__":
    unittest.main()
