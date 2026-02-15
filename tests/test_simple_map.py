import unittest

from yolozu.simple_map import evaluate_map


class TestSimpleMap(unittest.TestCase):
    def test_distillation_improves_map(self):
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

        student_map = evaluate_map(records, student, iou_thresholds=[0.5]).map50
        teacher_map = evaluate_map(records, teacher, iou_thresholds=[0.5]).map50
        self.assertGreaterEqual(teacher_map, student_map)

    def test_map_matches_windows_style_prediction_image(self):
        records = [
            {
                "image": "/dataset/images/img1.jpg",
                "labels": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
            }
        ]
        preds = [
            {
                "image": r"C:\dataset\images\img1.jpg",
                "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
            }
        ]
        result = evaluate_map(records, preds, iou_thresholds=[0.5])
        self.assertAlmostEqual(float(result.map50), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
