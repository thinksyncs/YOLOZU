import unittest

from yolozu.intrinsics import parse_intrinsics


class TestParseIntrinsics(unittest.TestCase):
    def test_dict_fx_fy_cx_cy(self):
        intr = parse_intrinsics({"fx": 100.0, "fy": 110.0, "cx": 50.0, "cy": 60.0})
        self.assertEqual(intr, {"fx": 100.0, "fy": 110.0, "cx": 50.0, "cy": 60.0})

    def test_list_fx_fy_cx_cy(self):
        intr = parse_intrinsics([100.0, 110.0, 50.0, 60.0])
        self.assertEqual(intr, {"fx": 100.0, "fy": 110.0, "cx": 50.0, "cy": 60.0})

    def test_k_3x3_nested_list(self):
        intr = parse_intrinsics([[100.0, 0.0, 50.0], [0.0, 110.0, 60.0], [0.0, 0.0, 1.0]])
        self.assertEqual(intr, {"fx": 100.0, "fy": 110.0, "cx": 50.0, "cy": 60.0})

    def test_k_3x3_flat_row_major(self):
        intr = parse_intrinsics([100.0, 0.0, 50.0, 0.0, 110.0, 60.0, 0.0, 0.0, 1.0])
        self.assertEqual(intr, {"fx": 100.0, "fy": 110.0, "cx": 50.0, "cy": 60.0})

    def test_opencv_matrix_dict(self):
        intr = parse_intrinsics(
            {
                "rows": 3,
                "cols": 3,
                "dt": "d",
                "data": [100.0, 0.0, 50.0, 0.0, 110.0, 60.0, 0.0, 0.0, 1.0],
            }
        )
        self.assertEqual(intr, {"fx": 100.0, "fy": 110.0, "cx": 50.0, "cy": 60.0})

    def test_nested_camera_matrix(self):
        intr = parse_intrinsics(
            {
                "camera_matrix": {
                    "rows": 3,
                    "cols": 3,
                    "data": [100.0, 0.0, 50.0, 0.0, 110.0, 60.0, 0.0, 0.0, 1.0],
                },
                "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0, 0, 0, 0, 0]},
            }
        )
        self.assertEqual(intr, {"fx": 100.0, "fy": 110.0, "cx": 50.0, "cy": 60.0})

    def test_rejects_unknown(self):
        self.assertIsNone(parse_intrinsics({"foo": 1}))


if __name__ == "__main__":
    unittest.main()

