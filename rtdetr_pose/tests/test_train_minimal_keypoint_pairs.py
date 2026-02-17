import unittest

from rtdetr_pose.train_minimal import _derive_keypoint_flip_pairs


class TestTrainMinimalKeypointPairs(unittest.TestCase):
    def test_derive_left_right_pairs(self):
        names = ["nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder", "tail"]
        pairs = _derive_keypoint_flip_pairs(names)
        self.assertEqual(pairs, [[1, 2], [3, 4]])

    def test_derive_l_r_prefix_pairs(self):
        names = ["kp0", "l_wrist", "r_wrist", "l_ankle", "r_ankle"]
        pairs = _derive_keypoint_flip_pairs(names)
        self.assertEqual(pairs, [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
