from __future__ import annotations

import math
from typing import Any, Iterable

from .boxes import iou_cxcywh_norm_dict
from .simple_map import evaluate_map


def _clip01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _fractal_score_transform(score: float, weight: float) -> float:
    score = _clip01(float(score))
    if weight <= 0:
        return score
    transformed = 1.0 - math.pow(max(0.0, 1.0 - score), float(weight))
    return _clip01(transformed)


def class_frequency_counts(records: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in records:
        for label in record.get("labels", []) or []:
            try:
                class_id = int(label.get("class_id", 0))
            except Exception:
                continue
            counts[class_id] = int(counts.get(class_id, 0) + 1)
    return counts


def class_frequency_counts_instance_segmentation(
    records: list[dict[str, Any]],
    *,
    allow_rgb_masks: bool = False,
) -> tuple[dict[int, int], list[str]]:
    from .instance_segmentation_eval import extract_gt_instances_from_record

    counts: dict[int, int] = {}
    warnings: list[str] = []
    for record in records:
        instances, wrn = extract_gt_instances_from_record(record, allow_rgb_masks=allow_rgb_masks)
        if wrn:
            warnings.extend([str(w) for w in wrn])
        for instance in instances:
            try:
                class_id = int(instance.get("class_id", 0))
            except Exception:
                continue
            counts[class_id] = int(counts.get(class_id, 0) + 1)
    return counts, warnings


def build_fracal_stats(
    records: list[dict[str, Any]],
    *,
    task: str = "bbox",
    allow_rgb_masks: bool = False,
) -> dict[str, Any]:
    task_norm = str(task or "bbox").strip().lower()
    warnings: list[str] = []
    if task_norm == "bbox":
        counts = class_frequency_counts(records)
    elif task_norm == "seg":
        counts, warnings = class_frequency_counts_instance_segmentation(records, allow_rgb_masks=allow_rgb_masks)
    else:
        raise ValueError("task must be one of: bbox, seg")

    return {
        "schema_version": 1,
        "method": "fracal",
        "task": task_norm,
        "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "summary": {
            "records_scanned": int(len(records)),
            "classes": int(len(counts)),
            "instances_total": int(sum(int(v) for v in counts.values())),
        },
        "warnings": list(warnings),
    }


def _coerce_class_counts(class_counts: dict[Any, Any] | None) -> dict[int, int]:
    if not isinstance(class_counts, dict):
        return {}
    out: dict[int, int] = {}
    for key, value in class_counts.items():
        try:
            class_id = int(key)
            count = int(value)
        except Exception:
            continue
        if count <= 0:
            continue
        out[class_id] = count
    return out


def _build_fracal_class_weights(class_counts: dict[int, int], *, alpha: float) -> dict[int, float]:
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    counts = {int(k): int(v) for k, v in class_counts.items() if int(v) > 0}
    if not counts:
        return {}

    max_count = max([int(v) for v in counts.values()] + [1])
    eps = 1e-9
    raw_weights: dict[int, float] = {}
    for class_id, count in counts.items():
        ratio = (float(max_count) + eps) / (float(count) + eps)
        raw_weights[int(class_id)] = float(math.pow(ratio, float(alpha)))

    geo_mean = 1.0
    if raw_weights:
        logs = [math.log(max(eps, float(v))) for v in raw_weights.values()]
        geo_mean = math.exp(sum(logs) / float(len(logs)))

    class_weights: dict[int, float] = {}
    for class_id, raw in raw_weights.items():
        class_weights[int(class_id)] = float(raw / max(eps, geo_mean))
    return class_weights


def _fracal_apply_score(
    *,
    score_orig: float,
    class_id: int,
    class_weights: dict[int, float],
    strength: float,
    min_score: float | None,
    max_score: float | None,
) -> float:
    score_orig = _clip01(float(score_orig))
    weight = float(class_weights.get(int(class_id), 1.0))
    score_fractal = _fractal_score_transform(score_orig, weight)
    score_new = (1.0 - float(strength)) * score_orig + float(strength) * score_fractal
    if min_score is not None:
        score_new = max(float(min_score), score_new)
    if max_score is not None:
        score_new = min(float(max_score), score_new)
    return _clip01(score_new)


def build_frequency_bins(
    class_counts: dict[int, int],
    *,
    head_fraction: float = 0.33,
    medium_fraction: float = 0.67,
) -> dict[int, str]:
    if not class_counts:
        return {}
    pairs = sorted(class_counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
    n = len(pairs)

    head_n = max(1, int(math.ceil(float(head_fraction) * float(n))))
    medium_n = max(head_n, int(math.ceil(float(medium_fraction) * float(n))))
    medium_n = min(medium_n, n)

    out: dict[int, str] = {}
    for idx, (class_id, _) in enumerate(pairs):
        if idx < head_n:
            out[int(class_id)] = "head"
        elif idx < medium_n:
            out[int(class_id)] = "medium"
        else:
            out[int(class_id)] = "tail"
    return out


def fracal_calibrate_predictions(
    records: list[dict[str, Any]],
    predictions_entries: list[dict[str, Any]],
    *,
    alpha: float = 0.5,
    strength: float = 1.0,
    min_score: float | None = None,
    max_score: float | None = None,
    class_counts: dict[int, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in [0, 1]")

    counts = _coerce_class_counts(class_counts) if class_counts is not None else class_frequency_counts(records)
    class_weights = _build_fracal_class_weights(counts, alpha=float(alpha))

    calibrated: list[dict[str, Any]] = []
    total_dets = 0
    changed_dets = 0
    for entry in predictions_entries:
        dets_out: list[dict[str, Any]] = []
        for det in entry.get("detections", []) or []:
            det_out = dict(det)
            total_dets += 1
            class_id = int(det.get("class_id", -1))
            score_orig = _clip01(float(det.get("score", 0.0)))
            score_new = _fracal_apply_score(
                score_orig=score_orig,
                class_id=class_id,
                class_weights=class_weights,
                strength=float(strength),
                min_score=min_score,
                max_score=max_score,
            )
            if abs(score_new - score_orig) > 1e-12:
                changed_dets += 1
            det_out["score"] = float(score_new)
            dets_out.append(det_out)
        calibrated.append({"image": entry.get("image"), "detections": dets_out})

    report = {
        "method": "fracal",
        "config": {
            "alpha": float(alpha),
            "strength": float(strength),
            "min_score": (None if min_score is None else float(min_score)),
            "max_score": (None if max_score is None else float(max_score)),
        },
        "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "class_weights": {str(k): float(v) for k, v in sorted(class_weights.items())},
        "summary": {
            "detections_total": int(total_dets),
            "detections_changed": int(changed_dets),
        },
    }
    return calibrated, report


def fracal_calibrate_instance_segmentation(
    records: list[dict[str, Any]],
    predictions_entries: list[dict[str, Any]],
    *,
    alpha: float = 0.5,
    strength: float = 1.0,
    min_score: float | None = None,
    max_score: float | None = None,
    class_counts: dict[int, int] | None = None,
    allow_rgb_masks: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in [0, 1]")

    warnings: list[str] = []
    if class_counts is not None:
        counts = _coerce_class_counts(class_counts)
    else:
        counts, warnings = class_frequency_counts_instance_segmentation(records, allow_rgb_masks=allow_rgb_masks)
    class_weights = _build_fracal_class_weights(counts, alpha=float(alpha))

    calibrated: list[dict[str, Any]] = []
    total_instances = 0
    changed_instances = 0

    for entry in predictions_entries:
        if not isinstance(entry, dict):
            continue
        out_entry = dict(entry)
        insts_out: list[dict[str, Any]] = []
        for inst in entry.get("instances", []) or []:
            if not isinstance(inst, dict):
                continue
            inst_out = dict(inst)
            total_instances += 1
            class_id = int(inst.get("class_id", -1))
            score_orig = _clip01(float(inst.get("score", 1.0)))
            score_new = _fracal_apply_score(
                score_orig=score_orig,
                class_id=class_id,
                class_weights=class_weights,
                strength=float(strength),
                min_score=min_score,
                max_score=max_score,
            )
            if abs(score_new - score_orig) > 1e-12:
                changed_instances += 1
            inst_out["score"] = float(score_new)
            insts_out.append(inst_out)
        out_entry["instances"] = insts_out
        calibrated.append(out_entry)

    report = {
        "method": "fracal",
        "task": "seg",
        "config": {
            "alpha": float(alpha),
            "strength": float(strength),
            "min_score": (None if min_score is None else float(min_score)),
            "max_score": (None if max_score is None else float(max_score)),
            "allow_rgb_masks": bool(allow_rgb_masks),
        },
        "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "class_weights": {str(k): float(v) for k, v in sorted(class_weights.items())},
        "summary": {
            "instances_total": int(total_instances),
            "instances_changed": int(changed_instances),
        },
        "warnings": list(warnings),
    }
    return calibrated, report


def _group_gt(records: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    by_image: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for record in records:
        image = str(record.get("image", ""))
        if not image:
            continue
        cls_map = by_image.setdefault(image, {})
        for label in record.get("labels", []) or []:
            try:
                class_id = int(label.get("class_id", 0))
            except Exception:
                continue
            cls_map.setdefault(class_id, []).append(label)
    return by_image


def _group_preds(predictions_entries: Iterable[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    by_image: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for entry in predictions_entries:
        image = str(entry.get("image", ""))
        if not image:
            continue
        cls_map = by_image.setdefault(image, {})
        for det in entry.get("detections", []) or []:
            if not isinstance(det, dict) or "bbox" not in det:
                continue
            try:
                class_id = int(det.get("class_id", 0))
            except Exception:
                class_id = 0
            cls_map.setdefault(class_id, []).append(det)
    return by_image


def _class_recall_at_iou(
    gt_by_image: dict[str, dict[int, list[dict[str, Any]]]],
    pred_by_image: dict[str, dict[int, list[dict[str, Any]]]],
    *,
    class_id: int,
    iou_thresh: float,
    max_detections: int,
) -> float:
    gt_total = 0
    tp = 0
    images = set(gt_by_image.keys()) | set(pred_by_image.keys())
    for image in images:
        gt_boxes = list((gt_by_image.get(image, {}) or {}).get(class_id, []) or [])
        gt_total += len(gt_boxes)
        if not gt_boxes:
            continue
        used = [False] * len(gt_boxes)
        preds = list((pred_by_image.get(image, {}) or {}).get(class_id, []) or [])
        preds.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        if max_detections > 0:
            preds = preds[: int(max_detections)]

        for pred in preds:
            bbox = pred.get("bbox")
            if not isinstance(bbox, dict):
                continue
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if used[idx]:
                    continue
                iou = iou_cxcywh_norm_dict(bbox, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= float(iou_thresh):
                used[best_idx] = True
                tp += 1

    if gt_total <= 0:
        return 0.0
    return float(tp) / float(gt_total)


def _collect_confidence_targets(
    gt_by_image: dict[str, dict[int, list[dict[str, Any]]]],
    pred_by_image: dict[str, dict[int, list[dict[str, Any]]]],
    *,
    iou_thresh: float,
    max_detections: int,
) -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    images = set(gt_by_image.keys()) | set(pred_by_image.keys())
    for image in images:
        gt_cls = gt_by_image.get(image, {}) or {}
        pred_cls = pred_by_image.get(image, {}) or {}

        used_map: dict[int, list[bool]] = {int(cid): [False] * len(gt_boxes) for cid, gt_boxes in gt_cls.items()}
        all_preds: list[dict[str, Any]] = []
        for _, dets in pred_cls.items():
            all_preds.extend(list(dets or []))
        all_preds.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        if max_detections > 0:
            all_preds = all_preds[: int(max_detections)]

        for pred in all_preds:
            score = _clip01(float(pred.get("score", 0.0)))
            try:
                class_id = int(pred.get("class_id", 0))
            except Exception:
                class_id = 0
            gt_boxes = list(gt_cls.get(class_id, []) or [])
            used = used_map.setdefault(class_id, [False] * len(gt_boxes))
            bbox = pred.get("bbox")
            correct = 0
            if isinstance(bbox, dict) and gt_boxes:
                best_iou = 0.0
                best_idx = -1
                for idx, gt in enumerate(gt_boxes):
                    if used[idx]:
                        continue
                    iou = iou_cxcywh_norm_dict(bbox, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx >= 0 and best_iou >= float(iou_thresh):
                    used[best_idx] = True
                    correct = 1
            pairs.append((score, int(correct)))
    return pairs


def _calibration_metrics(
    score_targets: list[tuple[float, int]],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    bins = max(1, int(bins))
    if not score_targets:
        return {
            "ece": 0.0,
            "confidence_bias": 0.0,
            "avg_confidence": 0.0,
            "avg_accuracy": 0.0,
            "bin_count": int(bins),
            "bins": [],
        }

    groups: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for score, target in score_targets:
        idx = min(bins - 1, max(0, int(math.floor(_clip01(score) * float(bins)))))
        groups[idx].append((float(score), int(target)))

    total = float(len(score_targets))
    ece = 0.0
    avg_conf = sum(s for s, _ in score_targets) / total
    avg_acc = sum(t for _, t in score_targets) / total
    out_bins: list[dict[str, Any]] = []
    for idx, bucket in enumerate(groups):
        n = len(bucket)
        if n <= 0:
            out_bins.append(
                {
                    "index": int(idx),
                    "score_range": [float(idx) / float(bins), float(idx + 1) / float(bins)],
                    "count": 0,
                    "confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue
        conf = sum(s for s, _ in bucket) / float(n)
        acc = sum(t for _, t in bucket) / float(n)
        gap = abs(conf - acc)
        ece += (float(n) / total) * gap
        out_bins.append(
            {
                "index": int(idx),
                "score_range": [float(idx) / float(bins), float(idx + 1) / float(bins)],
                "count": int(n),
                "confidence": float(conf),
                "accuracy": float(acc),
                "gap": float(gap),
            }
        )

    return {
        "ece": float(ece),
        "confidence_bias": float(avg_conf - avg_acc),
        "avg_confidence": float(avg_conf),
        "avg_accuracy": float(avg_acc),
        "bin_count": int(bins),
        "bins": out_bins,
    }


def evaluate_long_tail_detection(
    records: list[dict[str, Any]],
    predictions_entries: list[dict[str, Any]],
    *,
    iou_thresholds: Iterable[float] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10)),
    recall_iou: float = 0.5,
    max_detections: int = 100,
    head_fraction: float = 0.33,
    medium_fraction: float = 0.67,
    calibration_bins: int = 10,
    calibration_iou: float = 0.5,
) -> dict[str, Any]:
    thresholds = [float(v) for v in iou_thresholds]
    if not thresholds:
        thresholds = [0.5]

    map_result = evaluate_map(records, predictions_entries, iou_thresholds=thresholds)
    class_counts = class_frequency_counts(records)
    bins_map = build_frequency_bins(
        class_counts,
        head_fraction=float(head_fraction),
        medium_fraction=float(medium_fraction),
    )

    gt_by_image = _group_gt(records)
    pred_by_image = _group_preds(predictions_entries)
    all_classes = sorted(set(class_counts.keys()) | set(map_result.per_class.keys()))

    per_class_rows: list[dict[str, Any]] = []
    bucket_rows: dict[str, list[dict[str, Any]]] = {"head": [], "medium": [], "tail": [], "unknown": []}
    for class_id in all_classes:
        ap_map = dict(map_result.per_class.get(int(class_id), {}) or {})
        ap50 = float(ap_map.get("ap@0.50", 0.0))
        ap_values = [float(v) for v in ap_map.values()] if ap_map else []
        ap_mean = float(sum(ap_values) / float(len(ap_values))) if ap_values else 0.0

        recalls = [
            _class_recall_at_iou(
                gt_by_image,
                pred_by_image,
                class_id=int(class_id),
                iou_thresh=float(th),
                max_detections=int(max_detections),
            )
            for th in thresholds
        ]
        ar50 = _class_recall_at_iou(
            gt_by_image,
            pred_by_image,
            class_id=int(class_id),
            iou_thresh=float(recall_iou),
            max_detections=int(max_detections),
        )
        ar_mean = float(sum(recalls) / float(len(recalls))) if recalls else 0.0

        row = {
            "class_id": int(class_id),
            "count": int(class_counts.get(int(class_id), 0)),
            "frequency_bin": str(bins_map.get(int(class_id), "unknown")),
            "ap50": float(ap50),
            "ap50_95": float(ap_mean),
            "ar50": float(ar50),
            "ar50_95": float(ar_mean),
        }
        per_class_rows.append(row)
        bucket_rows.setdefault(row["frequency_bin"], []).append(row)

    def _mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
        if not rows:
            return None
        return float(sum(float(r.get(key, 0.0)) for r in rows) / float(len(rows)))

    freq_bins_out: dict[str, dict[str, Any]] = {}
    for bucket in ("head", "medium", "tail", "unknown"):
        rows = bucket_rows.get(bucket, [])
        freq_bins_out[bucket] = {
            "classes": int(len(rows)),
            "instances": int(sum(int(r.get("count", 0)) for r in rows)),
            "ap50": _mean_of(rows, "ap50"),
            "ap50_95": _mean_of(rows, "ap50_95"),
            "ar50": _mean_of(rows, "ar50"),
            "ar50_95": _mean_of(rows, "ar50_95"),
        }

    cal_pairs = _collect_confidence_targets(
        gt_by_image,
        pred_by_image,
        iou_thresh=float(calibration_iou),
        max_detections=int(max_detections),
    )
    calibration = _calibration_metrics(cal_pairs, bins=int(calibration_bins))

    macro = {
        "classes": int(len(per_class_rows)),
        "ap50": _mean_of(per_class_rows, "ap50") or 0.0,
        "ap50_95": _mean_of(per_class_rows, "ap50_95") or 0.0,
        "ar50": _mean_of(per_class_rows, "ar50") or 0.0,
        "ar50_95": _mean_of(per_class_rows, "ar50_95") or 0.0,
    }

    return {
        "metrics": {
            "map50": float(map_result.map50),
            "map50_95": float(map_result.map50_95),
            "macro": macro,
            "calibration": calibration,
        },
        "frequency_bins": freq_bins_out,
        "per_class": per_class_rows,
        "config": {
            "iou_thresholds": [float(v) for v in thresholds],
            "recall_iou": float(recall_iou),
            "max_detections": int(max_detections),
            "head_fraction": float(head_fraction),
            "medium_fraction": float(medium_fraction),
            "calibration_bins": int(calibration_bins),
            "calibration_iou": float(calibration_iou),
        },
        "counts": {
            "images": int(len(records)),
            "predictions_images": int(len(predictions_entries)),
            "classes": int(len(per_class_rows)),
            "detections": int(sum(len(entry.get("detections", []) or []) for entry in predictions_entries)),
            "labels": int(sum(len(record.get("labels", []) or []) for record in records)),
        },
    }
