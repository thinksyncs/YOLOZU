from .constraints import apply_constraints
from .geometry import corrected_intrinsics, recover_translation


def evaluate_candidate(
    cfg,
    class_key,
    bbox_center,
    bbox_wh,
    offsets,
    z_pred,
    size_wh,
    k,
    k_delta,
    r_mat,
):
    k_prime = corrected_intrinsics(k, k_delta)
    t_xyz = recover_translation(bbox_center, offsets, z_pred, k_prime)
    constraints = apply_constraints(
        cfg,
        class_key=class_key,
        bbox_wh=bbox_wh,
        size_wh=size_wh,
        intrinsics_fx_fy=(k_prime[0], k_prime[1]),
        t_xyz=t_xyz,
        r_mat=r_mat,
        z_pred=z_pred,
    )
    return {"k_prime": k_prime, "t_xyz": t_xyz, "constraints": constraints}
