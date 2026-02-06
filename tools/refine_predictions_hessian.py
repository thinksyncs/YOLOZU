#!/usr/bin/env python3
"""CLI tool to refine predictions using Hessian-based optimization.

This tool applies per-detection iterative refinement to regression outputs
using the Gauss-Newton solver with optional ground truth supervision.

Example usage:
    # Refine depth only (with GT supervision if available)
    python tools/refine_predictions_hessian.py \\
        --predictions reports/predictions.json \\
        --output reports/predictions_refined.json \\
        --refine-depth

    # Refine depth with dataset GT
    python tools/refine_predictions_hessian.py \\
        --predictions reports/predictions.json \\
        --dataset data/coco128 \\
        --output reports/predictions_refined.json \\
        --refine-depth --refine-rotation

    # Adjust solver parameters
    python tools/refine_predictions_hessian.py \\
        --predictions reports/predictions.json \\
        --output reports/predictions_refined.json \\
        --refine-depth \\
        --max-iterations 10 \\
        --convergence-threshold 1e-5 \\
        --damping 1e-2
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolozu.calibration import HessianSolverConfig, refine_predictions_hessian


def load_predictions(path):
    """Load predictions from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    
    # Handle different prediction formats.
    if isinstance(data, dict):
        if "predictions" in data:
            predictions = data["predictions"]
        else:
            # Dictionary format: image -> detections.
            predictions = []
            for image, dets in data.items():
                predictions.append({"image": image, "detections": dets})
    else:
        predictions = data
    
    return predictions


def load_dataset_records(dataset_root, split="train"):
    """Load dataset records for ground truth supervision."""
    from yolozu.dataset import build_manifest
    
    records = build_manifest(dataset_root, split=split)
    return list(records)


def save_predictions(predictions, path, wrap=False):
    """Save predictions to JSON file."""
    data = {"predictions": predictions} if wrap else predictions
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Refine predictions using Hessian-based optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input/output.
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output refined predictions JSON file",
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset root for GT supervision (optional)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap output in {'predictions': [...]} format",
    )
    
    # Refinement options.
    parser.add_argument(
        "--refine-depth",
        action="store_true",
        help="Refine depth predictions",
    )
    parser.add_argument(
        "--refine-rotation",
        action="store_true",
        help="Refine rotation predictions",
    )
    parser.add_argument(
        "--refine-offsets",
        action="store_true",
        help="Refine center offset predictions",
    )
    
    # Solver parameters.
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum Gauss-Newton iterations (default: 5)",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-4,
        help="Convergence threshold for parameter updates (default: 1e-4)",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=1e-3,
        help="Levenberg-Marquardt damping factor (default: 1e-3)",
    )
    
    # Weights.
    parser.add_argument(
        "--w-depth",
        type=float,
        default=1.0,
        help="Weight for depth residuals (default: 1.0)",
    )
    parser.add_argument(
        "--w-rotation",
        type=float,
        default=1.0,
        help="Weight for rotation residuals (default: 1.0)",
    )
    parser.add_argument(
        "--w-offsets",
        type=float,
        default=1.0,
        help="Weight for offset regularization (default: 1.0)",
    )
    
    args = parser.parse_args()
    
    # Check that at least one refinement option is enabled.
    if not (args.refine_depth or args.refine_rotation or args.refine_offsets):
        parser.error("At least one of --refine-depth, --refine-rotation, or --refine-offsets must be specified")
    
    # Load predictions.
    print(f"Loading predictions from {args.predictions}...", file=sys.stderr)
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} prediction entries", file=sys.stderr)
    
    # Load dataset records if provided.
    records = None
    if args.dataset:
        print(f"Loading dataset records from {args.dataset} (split: {args.split})...", file=sys.stderr)
        records = load_dataset_records(args.dataset, split=args.split)
        print(f"Loaded {len(records)} dataset records", file=sys.stderr)
    
    # Build solver config.
    config = HessianSolverConfig(
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        damping=args.damping,
        refine_depth=args.refine_depth,
        refine_rotation=args.refine_rotation,
        refine_offsets=args.refine_offsets,
        w_depth=args.w_depth,
        w_rotation=args.w_rotation,
        w_offsets=args.w_offsets,
    )
    
    print("Refinement configuration:", file=sys.stderr)
    print(f"  Refine depth: {config.refine_depth}", file=sys.stderr)
    print(f"  Refine rotation: {config.refine_rotation}", file=sys.stderr)
    print(f"  Refine offsets: {config.refine_offsets}", file=sys.stderr)
    print(f"  Max iterations: {config.max_iterations}", file=sys.stderr)
    print(f"  Convergence threshold: {config.convergence_threshold}", file=sys.stderr)
    print(f"  Damping: {config.damping}", file=sys.stderr)
    
    # Refine predictions.
    print("Refining predictions...", file=sys.stderr)
    refined_predictions = refine_predictions_hessian(
        predictions,
        records=records,
        config=config,
    )
    
    # Count how many detections were refined.
    total_detections = 0
    refined_detections = 0
    converged_detections = 0
    
    for entry in refined_predictions:
        if isinstance(entry, dict):
            dets = entry.get("detections", [])
            if isinstance(dets, list):
                total_detections += len(dets)
                for det in dets:
                    if isinstance(det, dict) and "hessian_refinement" in det:
                        refined_detections += 1
                        if det["hessian_refinement"].get("converged"):
                            converged_detections += 1
    
    print(f"Refinement complete:", file=sys.stderr)
    print(f"  Total detections: {total_detections}", file=sys.stderr)
    print(f"  Refined: {refined_detections}", file=sys.stderr)
    print(f"  Converged: {converged_detections}", file=sys.stderr)
    if refined_detections > 0:
        print(f"  Convergence rate: {converged_detections / refined_detections * 100:.1f}%", file=sys.stderr)
    
    # Save refined predictions.
    print(f"Saving refined predictions to {args.output}...", file=sys.stderr)
    save_predictions(refined_predictions, args.output, wrap=args.wrap)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
