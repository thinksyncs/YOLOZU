import copy
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.config import default_runtime_config_path, load_constraints
from yolozu.constraints import apply_constraints
from yolozu.math3d import rotation_matrix_axis_angle


def main():
    cfg = load_constraints(default_runtime_config_path("constraints.yaml"))
    sample = {
        "class_key": "sample",
        "bbox_wh": (20.0, 20.0),
        "size_wh": (0.2, 0.2),
        "intrinsics_fx_fy": (500.0, 500.0),
        "t_xyz": (0.0, 0.0, -1.0),
        "r_mat": rotation_matrix_axis_angle([1.0, 0.0, 0.0], 0.7853981633974483),
        "z_pred": 2.0,
    }

    results = {}
    for name in ("depth_prior", "table_plane", "upright"):
        ablated = copy.deepcopy(cfg)
        ablated["enabled"] = {k: False for k in ablated.get("enabled", {})}
        ablated["enabled"][name] = True
        results[name] = apply_constraints(ablated, **sample)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
