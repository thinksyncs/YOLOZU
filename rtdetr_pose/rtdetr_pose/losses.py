try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from .model import rot6d_to_matrix


def geodesic_distance(r1, r2):
    if torch is None:
        raise RuntimeError("torch is required for geodesic_distance")
    rel = torch.matmul(r1.transpose(-1, -2), r2)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.acos(cos_theta)


def symmetry_rotation_loss(rot6d_pred, r_gt, sym_rots):
    if torch is None:
        raise RuntimeError("torch is required for symmetry_rotation_loss")
    r_pred = rot6d_to_matrix(rot6d_pred)
    losses = []
    for s in sym_rots:
        r_eq = torch.matmul(r_gt, s)
        losses.append(geodesic_distance(r_pred, r_eq))
    return torch.stack(losses, dim=0).min(dim=0).values.mean()


class Losses(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, outputs, targets):
        if torch is None:
            raise RuntimeError("torch is required for Losses")
        loss = self.l1(outputs["bbox"], targets["bbox"])
        return {"loss": loss}
