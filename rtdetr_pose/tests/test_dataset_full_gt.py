import unittest

from rtdetr_pose.dataset import extract_full_gt_targets


class TestDatasetFullGT(unittest.TestCase):
    def test_extract_full_gt_expands_per_instance(self):
        record = {
            "labels": [
                {"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                {"class_id": 1, "bbox": {"cx": 0.4, "cy": 0.6, "w": 0.1, "h": 0.1}},
            ],
            # per-image values
            "M": "mask.png",
            "D_obj": [[0.0, 1.0], [0.0, 1.0]],
            "R_gt": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t_gt": [0.0, 0.0, 1.0],
            "K_gt": {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0},
        }

        out = extract_full_gt_targets(record, num_instances=2)
        self.assertEqual(len(out["M"]), 2)
        self.assertEqual(len(out["D_obj"]), 2)
        self.assertEqual(out["M"][0], "mask.png")
        self.assertTrue(out["M_mask"][0])
        self.assertTrue(out["D_obj_mask"][1])

        # Pose is expanded to per-instance lists.
        self.assertEqual(len(out["R_gt"]), 2)
        self.assertEqual(len(out["t_gt"]), 2)
        # Intrinsics is canonicalized to 3x3.
        self.assertEqual(len(out["K_gt"]), 3)
        self.assertEqual(len(out["K_gt"][0]), 3)

    def test_extract_full_gt_accepts_per_instance_lists(self):
        record = {
            "labels": [
                {"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                {"class_id": 1, "bbox": {"cx": 0.4, "cy": 0.6, "w": 0.1, "h": 0.1}},
            ],
            "M": ["m0.png", None],
            "D_obj": [None, "d1.npy"],
        }
        out = extract_full_gt_targets(record, num_instances=2)
        self.assertEqual(out["M"], ["m0.png", None])
        self.assertEqual(out["D_obj"], [None, "d1.npy"])
        self.assertEqual(out["M_mask"], [True, False])
        self.assertEqual(out["D_obj_mask"], [False, True])


if __name__ == "__main__":
    unittest.main()
