def corrected_intrinsics(k, delta):
    fx, fy, cx, cy = k
    dfx, dfy, dcx, dcy = delta
    return (
        fx * (1.0 + dfx),
        fy * (1.0 + dfy),
        cx + dcx,
        cy + dcy,
    )


def recover_translation(bbox_center, offsets, z, k_prime):
    u, v = bbox_center
    du, dv = offsets
    fx, fy, cx, cy = k_prime
    u_prime = u + du
    v_prime = v + dv
    x = (u_prime - cx) / fx * z
    y = (v_prime - cy) / fy * z
    return (x, y, z)
