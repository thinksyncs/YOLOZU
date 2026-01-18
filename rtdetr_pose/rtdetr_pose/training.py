"""Training helpers for the RT-DETR pose scaffold."""

from __future__ import annotations

from typing import Dict, List

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .matcher import linear_sum_assignment


def build_query_aligned_targets(
    logits,
    bbox_pred,
    targets: List[Dict],
    num_queries: int,
    cost_cls: float = 1.0,
    cost_bbox: float = 5.0,
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

    neg_log_prob = -torch.log_softmax(logits, dim=-1)
    bbox_norm = bbox_pred.sigmoid()

    for b in range(batch):
        gt_labels = targets[b].get("gt_labels")
        gt_bbox = targets[b].get("gt_bbox")
        if gt_labels is None or gt_bbox is None or gt_labels.numel() == 0:
            labels_q = torch.full((num_queries,), -1, dtype=torch.long, device=logits.device)
            bbox_q = torch.zeros((num_queries, 4), dtype=torch.float32, device=logits.device)
            aligned_labels.append(labels_q)
            aligned_bbox.append(bbox_q)
            continue

        gt_labels = gt_labels.to(device=logits.device)
        gt_bbox = gt_bbox.to(device=logits.device)

        # (Q, M) -> (M, Q)
        cls_cost = neg_log_prob[b][:, gt_labels].transpose(0, 1)
        # (M, Q)
        bbox_cost = torch.abs(bbox_norm[b].unsqueeze(0) - gt_bbox.unsqueeze(1)).sum(dim=-1)
        cost = cost_cls * cls_cost + cost_bbox * bbox_cost

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().tolist())
        labels_q = torch.full((num_queries,), -1, dtype=torch.long, device=logits.device)
        bbox_q = torch.zeros((num_queries, 4), dtype=torch.float32, device=logits.device)
        for r, c in zip(row_ind, col_ind):
            if 0 <= c < num_queries:
                labels_q[c] = gt_labels[r]
                bbox_q[c] = gt_bbox[r]
        aligned_labels.append(labels_q)
        aligned_bbox.append(bbox_q)

    return {
        "labels": torch.stack(aligned_labels, dim=0),
        "bbox": torch.stack(aligned_bbox, dim=0),
        "bbox_norm": bbox_norm,
    }
