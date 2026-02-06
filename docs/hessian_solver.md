# Hessian Solver for Regression Heads

## Overview

The Hessian solver provides per-detection iterative refinement of regression head predictions (depth, rotation, offsets) using second-order optimization. Unlike the L-BFGS calibration which optimizes dataset-wide scale and intrinsics, this solver refines individual predictions using available geometric constraints and ground truth supervision.

## Motivation

Regression heads in RT-DETR predict continuous values:
- **log_z**: Object depth (log space)
- **rot6d**: 6D rotation representation
- **offsets**: Center offset corrections (Δu, Δv)
- **k_delta**: Intrinsics corrections (handled by GlobalKHead)

These predictions can benefit from iterative refinement when:
1. Ground truth supervision is available (e.g., during validation/testing)
2. Geometric constraints can be applied (e.g., plane, upright)
3. Initial predictions have systematic errors that can be corrected

## Algorithm

The solver uses **Gauss-Newton optimization** with **Levenberg-Marquardt damping**:

```
For each detection:
  For each iteration:
    1. Compute residuals (errors between prediction and target)
    2. Compute Jacobian (gradients) via automatic differentiation
    3. Solve: (J^T J + λI) Δp = -J^T r
    4. Update parameters: p ← p + Δp
    5. Check convergence
```

Key features:
- **Second-order optimization**: Uses curvature information (Hessian approximation) for faster convergence
- **Damping**: Levenberg-Marquardt damping prevents divergence and handles ill-conditioned problems
- **Automatic convergence**: Stops when parameter updates are below threshold
- **NaN-safe**: Detects numerical instability and stops early

## Usage

### Python API

```python
from yolozu.calibration import HessianSolverConfig, refine_predictions_hessian

# Configure solver
config = HessianSolverConfig(
    max_iterations=5,
    convergence_threshold=1e-4,
    damping=1e-3,
    refine_depth=True,
    refine_rotation=True,
    refine_offsets=False,
)

# Refine predictions (with optional GT supervision)
refined_predictions = refine_predictions_hessian(
    predictions,
    records=dataset_records,  # Optional: for GT supervision
    config=config,
)
```

### CLI Tool

```bash
# Refine depth predictions using ground truth from dataset
python tools/refine_predictions_hessian.py \
  --predictions reports/predictions.json \
  --dataset data/coco128 \
  --output reports/predictions_refined.json \
  --refine-depth

# Refine multiple outputs
python tools/refine_predictions_hessian.py \
  --predictions reports/predictions.json \
  --dataset data/coco128 \
  --output reports/predictions_refined.json \
  --refine-depth \
  --refine-rotation \
  --max-iterations 10

# Refine with custom solver parameters
python tools/refine_predictions_hessian.py \
  --predictions reports/predictions.json \
  --dataset data/coco128 \
  --output reports/predictions_refined.json \
  --refine-depth \
  --max-iterations 10 \
  --convergence-threshold 1e-5 \
  --damping 1e-2 \
  --w-depth 1.0
```

## Configuration

### HessianSolverConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | int | 5 | Maximum Gauss-Newton iterations |
| `convergence_threshold` | float | 1e-4 | Stop when parameter update norm is below this |
| `damping` | float | 1e-3 | Levenberg-Marquardt damping factor (0 = pure Gauss-Newton) |
| `refine_depth` | bool | True | Refine log_z predictions |
| `refine_rotation` | bool | True | Refine rot6d predictions |
| `refine_offsets` | bool | True | Refine center offset predictions |
| `use_geometric_constraints` | bool | False | Apply plane/upright constraints |
| `w_depth` | float | 1.0 | Weight for depth residuals |
| `w_rotation` | float | 1.0 | Weight for rotation residuals |
| `w_offsets` | float | 1.0 | Weight for offset regularization |
| `w_plane` | float | 0.1 | Weight for plane constraint (if enabled) |
| `w_upright` | float | 0.1 | Weight for upright constraint (if enabled) |

### Refinement Metadata

Each refined detection includes metadata:

```json
{
  "log_z": 0.693,
  "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
  "hessian_refinement": {
    "iterations": 3,
    "converged": true,
    "final_residual": 0.0001
  }
}
```

## Comparison with L-BFGS Calibration

| Feature | Hessian Solver | L-BFGS Calibration |
|---------|----------------|-------------------|
| **Scope** | Per-detection | Dataset-wide |
| **Parameters** | log_z, rot6d, offsets | depth scale, shared k_delta |
| **Supervision** | Optional GT per detection | Matched GT required |
| **Use case** | Fine-tuning individual predictions | Global calibration |
| **Speed** | Fast (few iterations) | Moderate (L-BFGS) |
| **When to use** | Post-inference refinement | Dataset calibration |

Recommended workflow:
1. **Training**: Train model with standard losses
2. **Calibration**: Use L-BFGS to calibrate dataset-wide scale/intrinsics
3. **Refinement**: Use Hessian solver for per-detection fine-tuning (optional)

## Examples

### Example 1: Depth Refinement

```python
# Prediction with noisy depth
detection = {
    "class_id": 0,
    "score": 0.9,
    "log_z": 0.79,  # ~2.2m (noisy)
}

# Ground truth depth: 2.0m
config = HessianSolverConfig(refine_depth=True, refine_rotation=False, refine_offsets=False)
refined = refine_detection_hessian(detection, config=config, gt_depth=2.0)

# Result: log_z ≈ 0.69 (~2.0m, closer to GT)
```

### Example 2: Rotation Refinement

```python
# Prediction with small rotation error
detection = {
    "class_id": 0,
    "score": 0.9,
    "rot6d": [0.99, 0.01, 0.0, -0.01, 0.99, 0.0],  # Small perturbation
}

# Ground truth: identity rotation
gt_rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

config = HessianSolverConfig(refine_depth=False, refine_rotation=True, refine_offsets=False)
refined = refine_detection_hessian(detection, config=config, gt_rotation=gt_rotation)

# Result: rot6d closer to [1, 0, 0, 0, 1, 0]
```

### Example 3: Batch Refinement with Dataset

```python
from yolozu.dataset import build_manifest

# Load predictions
predictions = load_predictions("reports/predictions.json")

# Load dataset for GT supervision
records = list(build_manifest("data/coco128", split="train"))

# Refine all predictions
config = HessianSolverConfig(refine_depth=True, refine_rotation=True)
refined = refine_predictions_hessian(predictions, records=records, config=config)

# Save refined predictions
save_predictions(refined, "reports/predictions_refined.json")
```

## Performance Considerations

- **Convergence**: Typically converges in 3-5 iterations for well-conditioned problems
- **Speed**: ~1-2ms per detection on CPU (depends on enabled refinements)
- **Numerical Stability**: Includes NaN detection and early stopping for ill-conditioned cases
- **Memory**: Minimal overhead (small Jacobian matrices for 1-8 parameters)

### When NOT to Use

- **Production inference**: Adds latency; use only if accuracy improvement justifies cost
- **Identity rotations**: Refinement at identity can be numerically unstable; skip if already optimal
- **No supervision**: Without GT or constraints, only offset regularization is applied (minimal benefit)

## Implementation Notes

- Uses PyTorch automatic differentiation for Jacobian computation
- Supports mixed refinement (e.g., depth only, depth + rotation)
- Gracefully handles missing or invalid parameters
- Compatible with existing predictions JSON schema
- Thread-safe (no shared state between detections)

## Future Extensions

Potential improvements:
- [ ] Support for translation refinement (requires intrinsics)
- [ ] Batch processing with shared constraints
- [ ] GPU acceleration for large batches
- [ ] Adaptive damping based on residual reduction
- [ ] Symmetry-aware rotation refinement
- [ ] Integration with test-time adaptation (TTA/TTT)

## References

- Gauss-Newton algorithm: [Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)
- Levenberg-Marquardt damping: [Wikipedia](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
- RT-DETR: [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- 6D rotation representation: [arXiv:1812.07035](https://arxiv.org/abs/1812.07035)
