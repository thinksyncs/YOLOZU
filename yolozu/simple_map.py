from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MapResult:
    map50: float
    map50_95: float
    per_class: dict[int, dict[str, float]]


def _bbox_iou_cxcywh_norm(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1 = float(a["cx"]) - float(a["w"]) / 2.0
    ay1 = float(a["cy"]) - float(a["h"]) / 2.0
    ax2 = float(a["cx"]) + float(a["w"]) / 2.0
    ay2 = float(a["cy"]) + float(a["h"]) / 2.0

    bx1 = float(b["cx"]) - float(b["w"]) / 2.0
    by1 = float(b["cy"]) - float(b["h"]) / 2.0
    bx2 = float(b["cx"]) + float(b["w"]) / 2.0
    by2 = float(b["cy"]) + float(b["h"]) / 2.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def _group_ground_truth(records: list[dict[str, Any]]) -> tuple[dict[str, dict[int, list[dict[str, Any]]]], set[int]]:
    gt_by_image: dict[str, dict[int, list[dict[str, Any]]]] = {}
    classes: set[int] = set()
    for record in records:
        image = str(record.get("image", ""))
        if not image:
            continue
        entry = gt_by_image.setdefault(image, {})
        base = image.split("/")[-1]
        if base and base not in gt_by_image:
            gt_by_image[base] = entry
        for label in record.get("labels", []) or []:
            try:
                class_id = int(label.get("class_id", 0))
            except Exception:
                continue
            classes.add(class_id)
            entry.setdefault(class_id, []).append(label)
    return gt_by_image, classes


def _group_predictions(predictions_entries: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[int]]:
    preds: list[dict[str, Any]] = []
    classes: set[int] = set()
    for entry in predictions_entries:
        image = str(entry.get("image", ""))
        if not image:
            continue
        for det in entry.get("detections", []) or []:
            if "bbox" not in det:
                continue
            class_id = int(det.get("class_id", 0))
            classes.add(class_id)
            preds.append(
                {
                    "image": image,
                    "class_id": class_id,
                    "score": float(det.get("score", 0.0)),
                    "bbox": det.get("bbox"),
                }
            )
    return preds, classes


def _compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls or not precisions:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 1, 0, -1):
        if mpre[i - 1] < mpre[i]:
            mpre[i - 1] = mpre[i]

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return float(ap)


def _ap_for_class(
    *,
    preds: list[dict[str, Any]],
    gt_by_image: dict[str, dict[int, list[dict[str, Any]]]],
    images: list[str],
    class_id: int,
    iou_thresh: float,
) -> float:
    gt_count = 0
    gt_used: dict[str, list[bool]] = {}
    for image in images:
        boxes = gt_by_image.get(str(image), {}).get(class_id, [])
        used = [False] * len(boxes)
        gt_used[str(image)] = used
        gt_count += len(boxes)
        base = str(image).split("/")[-1]
        if base and base not in gt_used:
            gt_used[base] = used

    if gt_count == 0:
        return 0.0

    class_preds = [p for p in preds if p["class_id"] == class_id]
    class_preds.sort(key=lambda p: p["score"], reverse=True)

    tp: list[int] = []
    fp: list[int] = []

    for pred in class_preds:
        image = pred["image"]
        gt_boxes = gt_by_image.get(image, {}).get(class_id, [])
        used = gt_used.get(image, [])

        best_iou = 0.0
        best_idx = -1
        for idx, gt_box in enumerate(gt_boxes):
            iou = _bbox_iou_cxcywh_norm(pred["bbox"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_thresh and best_idx >= 0 and not used[best_idx]:
            used[best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    cum_tp = 0
    cum_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for i in range(len(tp)):
        cum_tp += tp[i]
        cum_fp += fp[i]
        recall = float(cum_tp) / float(max(1, gt_count))
        precision = float(cum_tp) / float(max(1, cum_tp + cum_fp))
        recalls.append(recall)
        precisions.append(precision)

    return _compute_ap(recalls, precisions)


def evaluate_map(
    records: list[dict[str, Any]],
    predictions_entries: Iterable[dict[str, Any]],
    *,
    iou_thresholds: Iterable[float] = (0.5,),
) -> MapResult:
    gt_by_image, gt_classes = _group_ground_truth(records)
    preds, pred_classes = _group_predictions(predictions_entries)
    classes = sorted(gt_classes.union(pred_classes))
    images = [str(r.get("image", "")) for r in records if str(r.get("image", ""))]

    thresholds = list(iou_thresholds)
    if not thresholds:
        thresholds = [0.5]

    per_class: dict[int, dict[str, float]] = {cid: {} for cid in classes}
    for thresh in thresholds:
        for cid in classes:
            ap = _ap_for_class(preds=preds, gt_by_image=gt_by_image, images=images, class_id=cid, iou_thresh=float(thresh))
            per_class[cid][f"ap@{thresh:.2f}"] = ap

    map50 = 0.0
    map50_95 = 0.0
    if classes:
        map50 = sum(per_class[cid].get("ap@0.50", 0.0) for cid in classes) / float(len(classes))
        map50_95 = sum(
            sum(per_class[cid].values()) / float(len(thresholds)) for cid in classes
        ) / float(len(classes))

    return MapResult(map50=float(map50), map50_95=float(map50_95), per_class=per_class)
