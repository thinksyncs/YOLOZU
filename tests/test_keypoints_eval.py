import unittest

from yolozu.keypoints_eval import evaluate_keypoints_pck


class TestKeypointsEval(unittest.TestCase):
    def test_evaluate_keypoints_matches_windows_style_index_key(self):
        records = [
            {
                "image": "/data/images/0001.jpg",
                "labels": [
                    {
                        "class_id": 0,
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 0.2,
                        "h": 0.2,
                        "keypoints": [0.5, 0.5, 2],
                    }
                ],
            }
        ]
        predictions_index = {
            r"C:\data\images\0001.jpg": [
                {
                    "class_id": 0,
                    "score": 0.9,
                    "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                    "keypoints": [0.5, 0.5, 2],
                }
            ]
        }

        res = evaluate_keypoints_pck(
            records=records,
            predictions_index=predictions_index,
            iou_threshold=0.5,
            pck_threshold=0.1,
            min_score=0.0,
        )
        self.assertEqual(int(res["metrics"]["instances_matched"]), 1)
        self.assertAlmostEqual(float(res["metrics"]["pck"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
