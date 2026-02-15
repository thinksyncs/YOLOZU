"""Tests for Hessian-based refinement solver."""

from __future__ import annotations

import math
import unittest

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

if torch is not None:
    from yolozu.calibration.hessian_solver import (
        HessianSolverConfig,
        refine_detection_hessian,
        refine_predictions_hessian,
    )


@unittest.skipIf(torch is None, "torch is not installed")
class TestHessianSolver(unittest.TestCase):
    def test_hessian_solver_config_defaults(self):
        config = HessianSolverConfig()
        self.assertEqual(config.max_iterations, 5)
        self.assertEqual(config.convergence_threshold, 1e-4)
        self.assertEqual(config.damping, 1e-3)
        self.assertTrue(config.refine_depth)
        self.assertTrue(config.refine_rotation)
        self.assertTrue(config.refine_offsets)

    def test_refine_detection_no_params(self):
        detection = {"class_id": 0, "score": 0.9}
        config = HessianSolverConfig()

        result = refine_detection_hessian(detection, config=config)
        self.assertEqual(result, detection)

    def test_refine_detection_depth_with_gt(self):
        gt_depth = 2.0
        noisy_log_z = math.log(2.2)  # 10% error.

        detection = {
            "class_id": 0,
            "score": 0.9,
            "log_z": noisy_log_z,
            "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "offsets": [0.0, 0.0],
        }

        config = HessianSolverConfig(
            max_iterations=10,
            refine_depth=True,
            refine_rotation=False,
            refine_offsets=False,
        )

        result = refine_detection_hessian(
            detection,
            config=config,
            gt_depth=gt_depth,
        )

        self.assertIn("log_z", result)
        refined_depth = math.exp(result["log_z"])

        initial_error = abs(math.exp(noisy_log_z) - gt_depth)
        refined_error = abs(refined_depth - gt_depth)
        self.assertLess(refined_error, initial_error)

        self.assertIn("hessian_refinement", result)
        self.assertGreater(result["hessian_refinement"]["iterations"], 0)

    def test_refine_detection_rotation_with_gt(self):
        gt_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        noisy_rot6d = [0.99, 0.01, 0.0, -0.01, 0.99, 0.0]
        detection = {
            "class_id": 0,
            "score": 0.9,
            "log_z": math.log(2.0),
            "rot6d": noisy_rot6d,
            "offsets": [0.0, 0.0],
        }

        config = HessianSolverConfig(
            max_iterations=10,
            refine_depth=False,
            refine_rotation=True,
            refine_offsets=False,
        )

        result = refine_detection_hessian(
            detection,
            config=config,
            gt_rotation=gt_rotation,
        )

        self.assertIn("rot6d", result)
        self.assertEqual(len(result["rot6d"]), 6)

        self.assertIn("hessian_refinement", result)
        self.assertGreater(result["hessian_refinement"]["iterations"], 0)

    def test_refine_detection_offsets(self):
        detection = {
            "class_id": 0,
            "score": 0.9,
            "log_z": math.log(2.0),
            "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "offsets": [10.0, 10.0],
        }

        config = HessianSolverConfig(
            max_iterations=10,
            refine_depth=False,
            refine_rotation=False,
            refine_offsets=True,
        )

        result = refine_detection_hessian(detection, config=config)

        self.assertIn("offsets", result)
        refined_norm = sum(x * x for x in result["offsets"]) ** 0.5
        original_norm = sum(x * x for x in detection["offsets"]) ** 0.5
        self.assertLess(refined_norm, original_norm)

    def test_refine_detection_convergence(self):
        detection = {
            "class_id": 0,
            "score": 0.9,
            "log_z": math.log(2.0),
        }

        config = HessianSolverConfig(
            max_iterations=20,
            convergence_threshold=1e-6,
            refine_depth=True,
        )

        result = refine_detection_hessian(
            detection,
            config=config,
            gt_depth=2.0,
        )

        self.assertIn("hessian_refinement", result)
        meta = result["hessian_refinement"]
        self.assertLessEqual(meta["iterations"], 5)

    def test_refine_detection_zero_max_iterations(self):
        detection = {
            "class_id": 0,
            "score": 0.9,
            "log_z": math.log(2.2),
        }

        config = HessianSolverConfig(
            max_iterations=0,
            refine_depth=True,
            refine_rotation=False,
            refine_offsets=False,
        )

        result = refine_detection_hessian(
            detection,
            config=config,
            gt_depth=2.0,
        )

        self.assertIn("hessian_refinement", result)
        self.assertEqual(int(result["hessian_refinement"]["iterations"]), 0)

    def test_refine_predictions_batch(self):
        predictions = [
            {
                "image": "test1.jpg",
                "detections": [
                    {
                        "class_id": 0,
                        "score": 0.9,
                        "log_z": math.log(2.2),
                        "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "offsets": [0.0, 0.0],
                    }
                ],
            },
            {
                "image": "test2.jpg",
                "detections": [
                    {
                        "class_id": 1,
                        "score": 0.8,
                        "log_z": math.log(1.8),
                        "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "offsets": [0.0, 0.0],
                    }
                ],
            },
        ]

        records = [
            {
                "image": "test1.jpg",
                "labels": [
                    {
                        "class_id": 0,
                        "t_gt": [0.0, 0.0, 2.0],
                        "R_gt": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    }
                ],
            },
            {
                "image": "test2.jpg",
                "labels": [
                    {
                        "class_id": 1,
                        "t_gt": [0.0, 0.0, 1.5],
                        "R_gt": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    }
                ],
            },
        ]

        config = HessianSolverConfig(
            max_iterations=5,
            refine_depth=True,
            refine_rotation=False,
            refine_offsets=False,
        )

        refined = refine_predictions_hessian(predictions, records, config=config)
        self.assertEqual(len(refined), 2)

        det1 = refined[0]["detections"][0]
        self.assertIn("log_z", det1)
        self.assertIn("hessian_refinement", det1)
        refined_depth1 = math.exp(det1["log_z"])
        self.assertLess(abs(refined_depth1 - 2.0), abs(2.2 - 2.0))

        det2 = refined[1]["detections"][0]
        self.assertIn("log_z", det2)
        self.assertIn("hessian_refinement", det2)
        refined_depth2 = math.exp(det2["log_z"])
        self.assertLess(abs(refined_depth2 - 1.5), abs(1.8 - 1.5))

    def test_refine_predictions_without_records(self):
        predictions = [
            {
                "image": "test.jpg",
                "detections": [
                    {
                        "class_id": 0,
                        "score": 0.9,
                        "log_z": math.log(2.0),
                        "offsets": [5.0, 5.0],
                    }
                ],
            }
        ]

        config = HessianSolverConfig(
            max_iterations=5,
            refine_depth=False,
            refine_offsets=True,
        )

        refined = refine_predictions_hessian(predictions, records=None, config=config)
        self.assertEqual(len(refined), 1)
        det = refined[0]["detections"][0]

        self.assertIn("offsets", det)
        refined_norm = sum(x * x for x in det["offsets"]) ** 0.5
        original_norm = sum(x * x for x in predictions[0]["detections"][0]["offsets"]) ** 0.5
        self.assertLess(refined_norm, original_norm)

    def test_refine_predictions_windows_style_path_key(self):
        predictions = [
            {
                "image": "test1.jpg",
                "detections": [
                    {
                        "class_id": 0,
                        "score": 0.9,
                        "log_z": math.log(2.2),
                    }
                ],
            }
        ]
        records = [
            {
                "image": r"C:\dataset\test1.jpg",
                "labels": [
                    {
                        "class_id": 0,
                        "t_gt": [0.0, 0.0, 2.0],
                    }
                ],
            }
        ]

        config = HessianSolverConfig(
            max_iterations=5,
            refine_depth=True,
            refine_rotation=False,
            refine_offsets=False,
        )

        refined = refine_predictions_hessian(predictions, records, config=config)
        det = refined[0]["detections"][0]
        self.assertIn("log_z", det)
        refined_depth = math.exp(det["log_z"])
        self.assertLess(abs(refined_depth - 2.0), abs(2.2 - 2.0))

    def test_refine_detection_invalid_inputs(self):
        detection = {
            "rot6d": [1.0, 0.0, 0.0],
        }

        config = HessianSolverConfig(refine_rotation=True)
        result = refine_detection_hessian(detection, config=config)
        self.assertEqual(result, detection)

        detection2 = {
            "offsets": [1.0],
        }

        config2 = HessianSolverConfig(refine_offsets=True)
        result2 = refine_detection_hessian(detection2, config=config2)
        self.assertEqual(result2, detection2)


if __name__ == "__main__":
    unittest.main()
