try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

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


def rotation_loss(rot6d_pred, r_gt):
    if torch is None:
        raise RuntimeError("torch is required for rotation_loss")
    r_pred = rot6d_to_matrix(rot6d_pred)
    return geodesic_distance(r_pred, r_gt).mean()


def log_depth_loss(log_z_pred, z_gt):
    if torch is None:
        raise RuntimeError("torch is required for log_depth_loss")
    log_z_gt = torch.log(torch.clamp(z_gt, min=1e-6))
    return torch.abs(log_z_pred - log_z_gt).mean()


def plane_loss(t_gt, plane):
    if torch is None:
        raise RuntimeError("torch is required for plane_loss")
    n = torch.tensor(plane["n"], device=t_gt.device, dtype=t_gt.dtype)
    d = torch.tensor(plane["d"], device=t_gt.device, dtype=t_gt.dtype)
    return torch.abs(torch.einsum("...i,i->...", t_gt, n) + d).mean()


def upright_loss(rot6d_pred, roll_range, pitch_range):
    if torch is None:
        raise RuntimeError("torch is required for upright_loss")
    r_pred = rot6d_to_matrix(rot6d_pred)
    roll = torch.atan2(r_pred[..., 2, 1], r_pred[..., 2, 2])
    pitch = torch.asin(torch.clamp(-r_pred[..., 2, 0], -1.0, 1.0))
    roll_min, roll_max = roll_range
    pitch_min, pitch_max = pitch_range
    roll_penalty = torch.clamp(roll - roll_max, min=0) + torch.clamp(roll_min - roll, min=0)
    pitch_penalty = torch.clamp(pitch - pitch_max, min=0) + torch.clamp(pitch_min - pitch, min=0)
    return (roll_penalty + pitch_penalty).mean()


class Losses(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = {
            "cls": 1.0,
            "box": 1.0,
            "z": 1.0,
            "rot": 1.0,
            "off": 1.0,
            "k": 1.0,
            "z_prior": 0.1,
            "plane": 0.1,
            "upright": 0.1,
        }
        if weights:
            self.weights.update(weights)
        self.l1 = nn.L1Loss()

    def forward(self, outputs, targets):
        if torch is None:
            raise RuntimeError("torch is required for Losses")
        losses = {}
        total = 0.0

        logits = outputs.get("logits")
        labels = targets.get("labels") or targets.get("class_gt")
        if logits is not None and labels is not None:
            loss_cls = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-1,
            )
            losses["loss_cls"] = loss_cls
            total = total + self.weights["cls"] * loss_cls

        bbox_pred = outputs.get("bbox")
        bbox_gt = targets.get("bbox")
        if bbox_pred is not None and bbox_gt is not None:
            loss_box = self.l1(bbox_pred, bbox_gt)
            losses["loss_box"] = loss_box
            total = total + self.weights["box"] * loss_box

        log_z_pred = outputs.get("log_z")
        z_gt = targets.get("z_gt") or targets.get("depth") or targets.get("z")
        if log_z_pred is not None and z_gt is not None:
            loss_z = log_depth_loss(log_z_pred, z_gt)
            losses["loss_z"] = loss_z
            total = total + self.weights["z"] * loss_z

        rot_pred = outputs.get("rot6d")
        r_gt = targets.get("R_gt") or targets.get("r_gt")
        sym_rots = targets.get("sym_rots")
        if rot_pred is not None and r_gt is not None:
            if sym_rots is not None:
                loss_rot = symmetry_rotation_loss(rot_pred, r_gt, sym_rots)
            else:
                loss_rot = rotation_loss(rot_pred, r_gt)
            losses["loss_rot"] = loss_rot
            total = total + self.weights["rot"] * loss_rot

        offsets_pred = outputs.get("offsets")
        offsets_gt = targets.get("offsets") or targets.get("offsets_gt")
        if offsets_pred is not None and offsets_gt is not None:
            loss_off = self.l1(offsets_pred, offsets_gt)
            losses["loss_off"] = loss_off
            total = total + self.weights["off"] * loss_off

        k_pred = outputs.get("k_delta")
        k_gt = targets.get("k_delta") or targets.get("k_delta_gt")
        if k_pred is not None and k_gt is not None:
            loss_k = self.l1(k_pred, k_gt)
            losses["loss_k"] = loss_k
            total = total + self.weights["k"] * loss_k

        z_prior = targets.get("z_prior")
        if log_z_pred is not None and z_prior is not None:
            loss_z_prior = self.l1(log_z_pred, torch.log(torch.clamp(z_prior, min=1e-6)))
            losses["loss_z_prior"] = loss_z_prior
            total = total + self.weights["z_prior"] * loss_z_prior

        t_gt = targets.get("t_gt")
        plane = targets.get("plane")
        if t_gt is not None and plane is not None:
            loss_plane = plane_loss(t_gt, plane)
            losses["loss_plane"] = loss_plane
            total = total + self.weights["plane"] * loss_plane

        upright = targets.get("upright_range")
        if rot_pred is not None and upright is not None:
            loss_upright = upright_loss(rot_pred, upright["roll"], upright["pitch"])
            losses["loss_upright"] = loss_upright
            total = total + self.weights["upright"] * loss_upright

        losses["loss"] = total
        return losses
