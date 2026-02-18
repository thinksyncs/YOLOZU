import unittest


class TestPoseEval(unittest.TestCase):
    def test_pose_success_counts_pairs_with_both_rot_and_trans(self):
        from yolozu.pose_eval import evaluate_pose

        record = {
            "image": "a.jpg",
            "labels": [
                {"class_id": 0, "cx": 0.25, "cy": 0.5, "w": 0.2, "h": 0.2},
                {"class_id": 0, "cx": 0.75, "cy": 0.5, "w": 0.2, "h": 0.2},
            ],
            "R_gt": [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            "t_gt": [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        }

        preds = [
            {
                "image": "a.jpg",
                "detections": [
                    {
                        "class_id": 0,
                        "score": 1.0,
                        "bbox": {"cx": 0.25, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "t_xyz": [0.0, 0.0, 1.0],
                    },
                    {
                        "class_id": 0,
                        "score": 1.0,
                        "bbox": {"cx": 0.75, "cy": 0.5, "w": 0.2, "h": 0.2},
                        # Rotation is present, translation is not: should not count toward pose_success denominator.
                        "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    },
                ],
            }
        ]

        r = evaluate_pose([record], preds, iou_threshold=0.5, min_score=0.0, success_rot_deg=15.0, success_trans=0.1)
        self.assertEqual(r.counts["gt_instances"], 2)
        self.assertEqual(r.counts["matches"], 2)
        self.assertEqual(r.counts["rot_measured"], 2)
        self.assertEqual(r.counts["trans_measured"], 1)
        self.assertEqual(r.counts["pose_measured"], 1)
        self.assertAlmostEqual(float(r.metrics["pose_success"]), 1.0, places=6)
        self.assertAlmostEqual(float(r.metrics["rot_success"]), 1.0, places=6)
        self.assertAlmostEqual(float(r.metrics["trans_success"]), 1.0, places=6)

    def test_pose_matches_windows_style_prediction_image(self):
        from yolozu.pose_eval import evaluate_pose

        record = {
            "image": "/data/images/0001.jpg",
            "labels": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
            "R_gt": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            "t_gt": [[0.0, 0.0, 1.0]],
        }
        preds = [
            {
                "image": r"C:\data\images\0001.jpg",
                "detections": [
                    {
                        "class_id": 0,
                        "score": 1.0,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "t_xyz": [0.0, 0.0, 1.0],
                    }
                ],
            }
        ]
        r = evaluate_pose([record], preds, iou_threshold=0.5, min_score=0.0)
        self.assertEqual(r.counts["matches"], 1)

    def test_add_and_adds_metrics_with_cad_points(self):
        from yolozu.pose_eval import evaluate_pose

        record = {
            "image": "a.jpg",
            "labels": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
            "R_gt": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            "t_gt": [[0.0, 0.0, 1.0]],
            "cad_points": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        }
        preds = [
            {
                "image": "a.jpg",
                "detections": [
                    {
                        "class_id": 0,
                        "score": 1.0,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "t_xyz": [0.1, 0.0, 1.0],
                    }
                ],
            }
        ]

        r = evaluate_pose([record], preds, iou_threshold=0.5, min_score=0.0)
        self.assertEqual(r.counts["add_measured"], 1)
        self.assertEqual(r.counts["adds_measured"], 1)
        self.assertAlmostEqual(float(r.metrics["add_mean"]), 0.1, places=6)
        self.assertAlmostEqual(float(r.metrics["adds_mean"]), 0.06666666666666667, places=6)


if __name__ == "__main__":
    unittest.main()
