import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.inference import infer_constraints


class TestInferenceConstraints(unittest.TestCase):
    def test_infer_constraints_happy_path(self):
        constraints_cfg = {
            "enabled": {"depth_prior": True, "table_plane": True, "upright": True},
            "depth_prior": {"default": {"min_z": 0.5, "max_z": 2.0}},
            "table_plane": {"n": [0.0, 0.0, 1.0], "d": 0.0},
            "upright": {"default": {"roll_deg": [-10.0, 10.0], "pitch_deg": [-10.0, 10.0]}},
        }
        entries = [
            {
                "image": "x.jpg",
                "image_size": {"width": 100, "height": 100},
                "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
                "size_wh": [0.2, 0.2],
                "detections": [
                    {
                        "class_id": 1,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "log_z": 0.0,
                        "offsets": [0.0, 0.0],
                        "k_delta": [0.0, 0.0, 0.0, 0.0],
                        "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    }
                ],
            }
        ]

        out = infer_constraints(entries, constraints_cfg=constraints_cfg, bbox_format="cxcywh_norm")
        det = out[0]["detections"][0]
        self.assertIn("t_xyz", det)
        self.assertIn("k_prime", det)
        self.assertIn("constraints", det)
        self.assertTrue(det["gate_ok"])

    def test_infer_constraints_missing_intrinsics(self):
        constraints_cfg = {"enabled": {"depth_prior": True}}
        entries = [
            {
                "image": "x.jpg",
                "image_size": {"width": 100, "height": 100},
                "detections": [
                    {
                        "class_id": 1,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "log_z": 0.0,
                        "offsets": [0.0, 0.0],
                    }
                ],
            }
        ]

        out = infer_constraints(entries, constraints_cfg=constraints_cfg, bbox_format="cxcywh_norm")
        det = out[0]["detections"][0]
        self.assertNotIn("t_xyz", det)
        self.assertNotIn("constraints", det)


if __name__ == "__main__":
    unittest.main()
