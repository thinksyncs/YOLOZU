import json
import math
import unittest
from pathlib import Path


def _has_torch():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipIf(not _has_torch(), "torch not installed")
class TestCalibratePredictionsLBFGS(unittest.TestCase):
    def test_recovers_depth_scale_from_t_gt_z(self):
        repo_root = Path(__file__).resolve().parents[1]

        # Create a tiny YOLO dataset in a temp dir.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "train2017").mkdir(parents=True)
            (root / "labels" / "train2017").mkdir(parents=True)

            img_path = root / "images" / "train2017" / "000001.jpg"
            img_path.write_bytes(b"")

            # One GT box.
            (root / "labels" / "train2017" / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            # Sidecar with pose + intrinsics + image size. z_gt is 2.0 (meters).
            sidecar = {
                "t_gt": [0.0, 0.0, 2.0],
                "K_gt": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
                "image_hw": [100, 100],
            }
            (root / "labels" / "train2017" / "000001.json").write_text(json.dumps(sidecar))

            # Prediction is off by 1000x (e.g., millimeters).
            z_pred = 2000.0
            log_z = math.log(z_pred)
            predictions = [
                {
                    "image": str(img_path),
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                            "log_z": log_z,
                            "offsets": [0.0, 0.0],
                            "k_delta": [0.0, 0.0, 0.0, 0.0],
                        }
                    ],
                }
            ]

            import sys

            sys.path.insert(0, str(repo_root))
            from yolozu.calibration import CalibConfig, calibrate_predictions_lbfgs
            from yolozu.dataset import build_manifest

            manifest = build_manifest(root, split="train2017")
            records = manifest["images"]

            calibrated, report = calibrate_predictions_lbfgs(records, predictions, config=CalibConfig())

            self.assertIsNotNone(report)
            self.assertGreaterEqual(report.matches, 1)
            # Expect s ~= 0.001.
            self.assertAlmostEqual(report.scale_s, 0.001, places=4)

            det = calibrated[0]["detections"][0]
            self.assertIn("log_z", det)
            z_after = math.exp(float(det["log_z"]))
            self.assertAlmostEqual(z_after, 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
