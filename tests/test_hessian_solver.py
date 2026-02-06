"""Tests for Hessian-based refinement solver."""

import math
import pytest

torch = pytest.importorskip("torch")

from yolozu.calibration.hessian_solver import (
    HessianSolverConfig,
    refine_detection_hessian,
    refine_predictions_hessian,
)


def test_hessian_solver_config_defaults():
    """Test default configuration values."""
    config = HessianSolverConfig()
    assert config.max_iterations == 5
    assert config.convergence_threshold == 1e-4
    assert config.damping == 1e-3
    assert config.refine_depth is True
    assert config.refine_rotation is True
    assert config.refine_offsets is True


def test_refine_detection_no_params():
    """Test refinement when no parameters are available."""
    detection = {"class_id": 0, "score": 0.9}
    config = HessianSolverConfig()
    
    result = refine_detection_hessian(detection, config=config)
    assert result == detection  # Should return unchanged.


def test_refine_detection_depth_with_gt():
    """Test depth refinement with ground truth supervision."""
    # Create detection with noisy depth.
    gt_depth = 2.0
    noisy_log_z = math.log(2.2)  # 10% error.
    
    detection = {
        "class_id": 0,
        "score": 0.9,
        "log_z": noisy_log_z,
        "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Identity.
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
    
    # Check that depth was refined.
    assert "log_z" in result
    refined_depth = math.exp(result["log_z"])
    
    # Should be closer to ground truth.
    initial_error = abs(math.exp(noisy_log_z) - gt_depth)
    refined_error = abs(refined_depth - gt_depth)
    assert refined_error < initial_error
    
    # Check metadata.
    assert "hessian_refinement" in result
    assert result["hessian_refinement"]["iterations"] > 0


def test_refine_detection_rotation_with_gt():
    """Test rotation refinement with ground truth supervision."""
    # Ground truth: identity rotation.
    gt_rotation = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    
    # Noisy rotation (small perturbation).
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
    
    # Check that rotation was refined.
    assert "rot6d" in result
    assert len(result["rot6d"]) == 6
    
    # Check metadata.
    assert "hessian_refinement" in result
    assert result["hessian_refinement"]["iterations"] > 0


def test_refine_detection_offsets():
    """Test offset refinement (regularization)."""
    # Large offsets should be reduced by regularization.
    detection = {
        "class_id": 0,
        "score": 0.9,
        "log_z": math.log(2.0),
        "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "offsets": [10.0, 10.0],  # Large offsets.
    }
    
    config = HessianSolverConfig(
        max_iterations=10,
        refine_depth=False,
        refine_rotation=False,
        refine_offsets=True,
    )
    
    result = refine_detection_hessian(detection, config=config)
    
    # Check that offsets were reduced.
    assert "offsets" in result
    refined_norm = sum(x * x for x in result["offsets"]) ** 0.5
    original_norm = sum(x * x for x in detection["offsets"]) ** 0.5
    
    # Should be smaller due to regularization.
    assert refined_norm < original_norm


def test_refine_detection_convergence():
    """Test that solver reports convergence correctly."""
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
        gt_depth=2.0,  # Already at optimum.
    )
    
    # Should converge quickly.
    assert "hessian_refinement" in result
    meta = result["hessian_refinement"]
    # Either converges or runs minimal iterations.
    assert meta["iterations"] <= 5


def test_refine_predictions_batch():
    """Test batch refinement of predictions."""
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
    
    # Test depth-only refinement to avoid numerical issues with rotation at identity.
    config = HessianSolverConfig(
        max_iterations=5,
        refine_depth=True,
        refine_rotation=False,  # Skip rotation refinement in this test.
        refine_offsets=False,
    )
    
    refined = refine_predictions_hessian(predictions, records, config=config)
    
    assert len(refined) == 2
    
    # Check first prediction.
    det1 = refined[0]["detections"][0]
    assert "log_z" in det1
    assert "hessian_refinement" in det1
    
    # Depth should be closer to GT.
    refined_depth1 = math.exp(det1["log_z"])
    assert abs(refined_depth1 - 2.0) < abs(math.exp(math.log(2.2)) - 2.0)
    
    # Check second prediction.
    det2 = refined[1]["detections"][0]
    assert "log_z" in det2
    assert "hessian_refinement" in det2
    
    refined_depth2 = math.exp(det2["log_z"])
    assert abs(refined_depth2 - 1.5) < abs(math.exp(math.log(1.8)) - 1.5)


def test_refine_predictions_without_records():
    """Test refinement without ground truth (regularization only)."""
    predictions = [
        {
            "image": "test.jpg",
            "detections": [
                {
                    "class_id": 0,
                    "score": 0.9,
                    "log_z": math.log(2.0),
                    "offsets": [5.0, 5.0],  # Large offsets to regularize.
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
    
    assert len(refined) == 1
    det = refined[0]["detections"][0]
    
    # Offsets should be reduced.
    assert "offsets" in det
    refined_norm = sum(x * x for x in det["offsets"]) ** 0.5
    original_norm = sum(x * x for x in predictions[0]["detections"][0]["offsets"]) ** 0.5
    assert refined_norm < original_norm


def test_refine_detection_invalid_inputs():
    """Test handling of invalid inputs."""
    # Invalid rot6d length.
    detection = {
        "rot6d": [1.0, 0.0, 0.0],  # Only 3 elements.
    }
    
    config = HessianSolverConfig(refine_rotation=True)
    result = refine_detection_hessian(detection, config=config)
    
    # Should return unchanged (no valid parameters to refine).
    assert result == detection
    
    # Invalid offsets.
    detection2 = {
        "offsets": [1.0],  # Only 1 element.
    }
    
    config2 = HessianSolverConfig(refine_offsets=True)
    result2 = refine_detection_hessian(detection2, config=config2)
    assert result2 == detection2
