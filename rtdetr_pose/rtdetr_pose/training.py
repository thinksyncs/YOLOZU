"""Training helpers for the RT-DETR pose scaffold."""

from __future__ import annotations

from typing import Dict, List

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .matcher import linear_sum_assignment


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _rot6d_to_matrix(x):
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def build_query_aligned_targets(
    logits,
    bbox_pred,
    targets: List[Dict],
    num_queries: int,
    cost_cls: float = 1.0,
    cost_bbox: float = 5.0,
    log_z_pred=None,
    rot6d_pred=None,
    cost_z: float = 0.0,
    cost_rot: float = 0.0,
):
    """Match queries to GT per-sample and build query-aligned targets.

    targets: list of dicts with keys gt_labels (M,) and gt_bbox (M,4)
    returns: dict with labels (B,Q) and bbox (B,Q,4) plus bbox_norm (B,Q,4)
    """

    if torch is None:
        raise RuntimeError("torch is required for build_query_aligned_targets")

    batch, queries, _ = logits.shape
    if queries != num_queries:
        raise ValueError("num_queries mismatch")

    aligned_labels = []
    aligned_bbox = []
    aligned_z = []
    aligned_r = []
    aligned_offsets = []
    aligned_mask = []

    neg_log_prob = -torch.log_softmax(logits, dim=-1)
    bbox_norm = bbox_pred.sigmoid()

    if log_z_pred is not None:
        log_z_pred = log_z_pred
    if rot6d_pred is not None:
        r_pred_all = _rot6d_to_matrix(rot6d_pred)
    else:
        r_pred_all = None

    for b in range(batch):
        gt_labels = targets[b].get("gt_labels")
        gt_bbox = targets[b].get("gt_bbox")
        gt_z = _first_present(targets[b], ("gt_z", "z_gt", "z", "depth"))
        gt_r = _first_present(targets[b], ("gt_R", "R_gt", "R", "gt_r"))
        gt_offsets = _first_present(targets[b], ("gt_offsets", "offsets", "offsets_gt"))
        if gt_labels is None or gt_bbox is None or gt_labels.numel() == 0:
            labels_q = torch.full((num_queries,), -1, dtype=torch.long, device=logits.device)
            bbox_q = torch.zeros((num_queries, 4), dtype=torch.float32, device=logits.device)
            mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
            z_q = torch.zeros((num_queries, 1), dtype=torch.float32, device=logits.device)
            r_q = torch.eye(3, device=logits.device, dtype=torch.float32).unsqueeze(0).repeat(
                num_queries, 1, 1
            )
            off_q = torch.zeros((num_queries, 2), dtype=torch.float32, device=logits.device)
            aligned_labels.append(labels_q)
            aligned_bbox.append(bbox_q)
            aligned_mask.append(mask_q)
            aligned_z.append(z_q)
            aligned_r.append(r_q)
            aligned_offsets.append(off_q)
            continue

        gt_labels = gt_labels.to(device=logits.device)
        gt_bbox = gt_bbox.to(device=logits.device)
        if gt_z is not None:
            gt_z = gt_z.to(device=logits.device, dtype=torch.float32)
            if gt_z.ndim == 1:
                gt_z = gt_z.unsqueeze(-1)
        if gt_r is not None:
            gt_r = gt_r.to(device=logits.device, dtype=torch.float32)
        if gt_offsets is not None:
            gt_offsets = gt_offsets.to(device=logits.device, dtype=torch.float32)

        # (Q, M) -> (M, Q)
        cls_cost = neg_log_prob[b][:, gt_labels].transpose(0, 1)
        # (M, Q)
        bbox_cost = torch.abs(bbox_norm[b].unsqueeze(0) - gt_bbox.unsqueeze(1)).sum(dim=-1)
        cost = cost_cls * cls_cost + cost_bbox * bbox_cost

        if cost_z and log_z_pred is not None and gt_z is not None:
            log_z_gt = torch.log(torch.clamp(gt_z, min=1e-6))
            z_cost = torch.abs(log_z_pred[b].unsqueeze(0) - log_z_gt.unsqueeze(1)).sum(dim=-1)
            cost = cost + float(cost_z) * z_cost

        if cost_rot and r_pred_all is not None and gt_r is not None:
            # Geodesic angle via trace(R_pred^T R_gt) == sum(R_pred * R_gt)
            trace = (r_pred_all[b].unsqueeze(0) * gt_r.unsqueeze(1)).sum(dim=(-1, -2))
            cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
            rot_cost = torch.acos(cos_theta)
            cost = cost + float(cost_rot) * rot_cost

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().tolist())
        labels_q = torch.full((num_queries,), -1, dtype=torch.long, device=logits.device)
        bbox_q = torch.zeros((num_queries, 4), dtype=torch.float32, device=logits.device)
        mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
        z_q = torch.zeros((num_queries, 1), dtype=torch.float32, device=logits.device)
        r_q = torch.eye(3, device=logits.device, dtype=torch.float32).unsqueeze(0).repeat(
            num_queries, 1, 1
        )
        off_q = torch.zeros((num_queries, 2), dtype=torch.float32, device=logits.device)
        for r, c in zip(row_ind, col_ind):
            if 0 <= c < num_queries:
                labels_q[c] = gt_labels[r]
                bbox_q[c] = gt_bbox[r]
                mask_q[c] = True
                if gt_z is not None:
                    z_q[c] = gt_z[r]
                if gt_r is not None:
                    r_q[c] = gt_r[r]
                if gt_offsets is not None:
                    off_q[c] = gt_offsets[r]
        aligned_labels.append(labels_q)
        aligned_bbox.append(bbox_q)
        aligned_mask.append(mask_q)
        aligned_z.append(z_q)
        aligned_r.append(r_q)
        aligned_offsets.append(off_q)

    return {
        "labels": torch.stack(aligned_labels, dim=0),
        "bbox": torch.stack(aligned_bbox, dim=0),
        "bbox_norm": bbox_norm,
        "mask": torch.stack(aligned_mask, dim=0),
        "z_gt": torch.stack(aligned_z, dim=0),
        "R_gt": torch.stack(aligned_r, dim=0),
        "offsets": torch.stack(aligned_offsets, dim=0),
    }
