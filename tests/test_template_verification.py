import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.template_verification import apply_template_verification


class TestTemplateVerification(unittest.TestCase):
    def test_apply_template_verification_topk(self):
        entries = [
            {
                "image": "x.jpg",
                "detections": [
                    {
                        "class_id": 1,
                        "score": 0.9,
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    {
                        "class_id": 1,
                        "score": 0.1,
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                ],
            }
        ]
        symmetry_map = {1: {"type": "C2", "axis": [0.0, 0.0, 1.0]}}

        def score_fn(det, rot):
            return float(rot[0][0] + rot[1][1] + rot[2][2])

        out = apply_template_verification(
            entries,
            symmetry_map=symmetry_map,
            score_fn=score_fn,
            top_k=1,
            sample_count=4,
        )
        dets = out[0]["detections"]
        self.assertIn("score_tmp_sym", dets[0])
        self.assertNotIn("score_tmp_sym", dets[1])
        self.assertAlmostEqual(dets[0]["score_tmp_sym"], 3.0)


if __name__ == "__main__":
    unittest.main()
