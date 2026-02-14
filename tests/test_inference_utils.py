import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.inference_utils import apply_inference_utilities


class TestInferenceUtilities(unittest.TestCase):
    def test_apply_inference_utilities(self):
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
                        "score": 0.9,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "log_z": 0.0,
                        "offsets": [0.0, 0.0],
                        "k_delta": [0.0, 0.0, 0.0, 0.0],
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    {
                        "class_id": 1,
                        "score": 0.1,
                        "bbox": {"cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1},
                        "log_z": 0.0,
                        "offsets": [0.0, 0.0],
                        "k_delta": [0.0, 0.0, 0.0, 0.0],
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                ],
            }
        ]
        symmetry_map = {1: {"type": "C2", "axis": [0.0, 0.0, 1.0]}}

        def score_fn(det, rot):
            return float(rot[0][0] + rot[1][1] + rot[2][2])

        out = apply_inference_utilities(
            entries,
            constraints_cfg=constraints_cfg,
            symmetry_map=symmetry_map,
            template_score_fn=score_fn,
            template_top_k=1,
        )
        dets = out[0]["detections"]
        self.assertTrue(dets[0]["gate_ok"])
        self.assertIn("score_tmp_sym", dets[0])
        self.assertNotIn("score_tmp_sym", dets[1])


if __name__ == "__main__":
    unittest.main()
