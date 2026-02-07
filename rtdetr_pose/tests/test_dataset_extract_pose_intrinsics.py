import unittest


class TestDatasetExtractPoseIntrinsics(unittest.TestCase):
    def test_broadcasts_pose_and_converts_K(self):
        from rtdetr_pose.dataset import extract_pose_intrinsics_targets

        record = {
            "pose": {"t": [1, 2, 3], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "intrinsics": [100.0, 200.0, 50.0, 60.0],
        }
        out = extract_pose_intrinsics_targets(record, num_instances=2)

        self.assertEqual(out["K_gt"], [[100.0, 0.0, 50.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]])
        self.assertEqual(out["t_gt"], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        self.assertEqual(out["R_gt"], [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ])

    def test_accepts_per_instance_t_list(self):
        from rtdetr_pose.dataset import extract_pose_intrinsics_targets

        record = {"t_gt": [[0, 0, 0.5], [0, 0, 0.8]]}
        out = extract_pose_intrinsics_targets(record, num_instances=2)
        self.assertEqual(out["t_gt"], [[0.0, 0.0, 0.5], [0.0, 0.0, 0.8]])

    def test_accepts_opencv_camera_matrix_dict(self):
        from rtdetr_pose.dataset import extract_pose_intrinsics_targets

        record = {
            "K_gt": {
                "camera_matrix": {
                    "rows": 3,
                    "cols": 3,
                    "dt": "d",
                    "data": [100.0, 0.0, 50.0, 0.0, 200.0, 60.0, 0.0, 0.0, 1.0],
                }
            }
        }
        out = extract_pose_intrinsics_targets(record, num_instances=1)
        self.assertEqual(out["K_gt"], [[100.0, 0.0, 50.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
