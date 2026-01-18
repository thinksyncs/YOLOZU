import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.config import load_constraints, load_symmetry_map
from yolozu.math3d import mat_identity, rotation_z
from yolozu.pipeline import evaluate_candidate
from yolozu.symmetry import min_symmetry_geodesic


def main():
    constraints = load_constraints(repo_root / "constraints.yaml")
    symmetry = load_symmetry_map(repo_root / "symmetry.json")

    class_key = "sample"
    sym_spec = symmetry.get(class_key, {"type": "none"})
    r_gt = mat_identity()
    r_pred = rotation_z(1.5707963267948966)
    loss = min_symmetry_geodesic(r_pred, r_gt, sym_spec)

    eval_result = evaluate_candidate(
        constraints,
        class_key=class_key,
        bbox_center=(320.0, 240.0),
        bbox_wh=(20.0, 20.0),
        offsets=(0.0, 0.0),
        z_pred=2.0,
        size_wh=(0.2, 0.2),
        k=(500.0, 500.0, 320.0, 240.0),
        k_delta=(0.0, 0.0, 0.0, 0.0),
        r_mat=r_gt,
    )

    payload = {"symmetry_loss": loss, "eval": eval_result}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
