import math


def symmetry_geodesic(r_pred, r_gt, sym_rots):
    def geodesic(a, b):
        trace = a[0][0] * b[0][0] + a[1][1] * b[1][1] + a[2][2] * b[2][2]
        cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
        return math.acos(cos_theta)

    best = None
    for s in sym_rots:
        r_eq = [
            [
                r_gt[0][0] * s[0][0] + r_gt[0][1] * s[1][0] + r_gt[0][2] * s[2][0],
                r_gt[0][0] * s[0][1] + r_gt[0][1] * s[1][1] + r_gt[0][2] * s[2][1],
                r_gt[0][0] * s[0][2] + r_gt[0][1] * s[1][2] + r_gt[0][2] * s[2][2],
            ],
            [
                r_gt[1][0] * s[0][0] + r_gt[1][1] * s[1][0] + r_gt[1][2] * s[2][0],
                r_gt[1][0] * s[0][1] + r_gt[1][1] * s[1][1] + r_gt[1][2] * s[2][1],
                r_gt[1][0] * s[0][2] + r_gt[1][1] * s[1][2] + r_gt[1][2] * s[2][2],
            ],
            [
                r_gt[2][0] * s[0][0] + r_gt[2][1] * s[1][0] + r_gt[2][2] * s[2][0],
                r_gt[2][0] * s[0][1] + r_gt[2][1] * s[1][1] + r_gt[2][2] * s[2][1],
                r_gt[2][0] * s[0][2] + r_gt[2][1] * s[1][2] + r_gt[2][2] * s[2][2],
            ],
        ]
        value = geodesic(r_pred, r_eq)
        if best is None or value < best:
            best = value
    return best if best is not None else 0.0
