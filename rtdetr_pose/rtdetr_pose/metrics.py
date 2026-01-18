import math

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _geodesic_matrix(a, b):
    trace = a[0][0] * b[0][0] + a[1][1] * b[1][1] + a[2][2] * b[2][2]
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.acos(cos_theta)


def geodesic_torch(r1, r2):
    rel = torch.matmul(r1.transpose(-1, -2), r2)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.acos(cos_theta)


def symmetry_geodesic(r_pred, r_gt, sym_rots):
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
        value = _geodesic_matrix(r_pred, r_eq)
        if best is None or value < best:
            best = value
    return best if best is not None else 0.0


def symmetry_geodesic_torch(r_pred, r_gt, sym_rots):
    if torch is None:
        raise RuntimeError("torch is required for symmetry_geodesic_torch")
    losses = []
    for s in sym_rots:
        losses.append(geodesic_torch(r_pred, torch.matmul(r_gt, s)))
    return torch.stack(losses, dim=0).min(dim=0).values


def add_s_torch(points, r_pred, t_pred, r_gt, t_gt, sym_rots=None):
    if torch is None:
        raise RuntimeError("torch is required for add_s_torch")
    p_pred = torch.matmul(points, r_pred.transpose(-1, -2)) + t_pred
    if sym_rots is None:
        p_gt = torch.matmul(points, r_gt.transpose(-1, -2)) + t_gt
        return torch.norm(p_pred - p_gt, dim=-1).mean()
    errors = []
    for s in sym_rots:
        r_eq = torch.matmul(r_gt, s)
        p_gt = torch.matmul(points, r_eq.transpose(-1, -2)) + t_gt
        errors.append(torch.norm(p_pred - p_gt, dim=-1).mean())
    return torch.stack(errors, dim=0).min(dim=0).values


def depth_error(z_pred, z_gt):
    if torch is not None and hasattr(z_pred, "shape"):
        return torch.abs(z_pred - z_gt).mean()
    return sum(abs(a - b) for a, b in zip(z_pred, z_gt)) / max(1, len(z_pred))
