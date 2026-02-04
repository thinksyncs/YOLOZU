from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .simple_map import _bbox_iou_cxcywh_norm


@dataclass(frozen=True)
class DistillStats:
    matched: int
    added: int
    avg_score_gap: float


def _index_by_image(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        image = str(entry.get("image", ""))
        if not image:
            continue
        dets = entry.get("detections", []) or []
        index[image] = list(dets) if isinstance(dets, list) else []
        base = image.split("/")[-1]
        if base and base not in index:
            index[base] = index[image]
    return index


def distill_predictions(
    student_entries: list[dict[str, Any]],
    teacher_entries: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.7,
    alpha: float = 0.5,
    add_missing: bool = True,
    add_score_scale: float = 0.5,
) -> tuple[list[dict[str, Any]], DistillStats]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if iou_threshold < 0.0:
        raise ValueError("iou_threshold must be >= 0")

    teacher_index = _index_by_image(teacher_entries)

    out_entries: list[dict[str, Any]] = []
    matched = 0
    added = 0
    total_gap = 0.0

    for entry in student_entries:
        image = str(entry.get("image", ""))
        if not image:
            continue
        student_dets = [dict(d) for d in (entry.get("detections", []) or [])]
        teacher_dets = teacher_index.get(image, []) or []
        teacher_used = [False] * len(teacher_dets)

        for det in student_dets:
            best_iou = 0.0
            best_idx = -1
            for idx, tdet in enumerate(teacher_dets):
                if int(tdet.get("class_id", -1)) != int(det.get("class_id", -2)):
                    continue
                iou = _bbox_iou_cxcywh_norm(det.get("bbox", {}), tdet.get("bbox", {}))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                teacher_used[best_idx] = True
                t_score = float(teacher_dets[best_idx].get("score", 0.0))
                s_score = float(det.get("score", 0.0))
                total_gap += abs(t_score - s_score)
                matched += 1
                det["score"] = max(s_score, alpha * t_score + (1.0 - alpha) * s_score)

        if add_missing:
            for idx, tdet in enumerate(teacher_dets):
                if teacher_used[idx]:
                    continue
                score = float(tdet.get("score", 0.0)) * float(add_score_scale)
                det_out = dict(tdet)
                det_out["score"] = score
                student_dets.append(det_out)
                added += 1

        out_entries.append({"image": image, "detections": student_dets})

    avg_gap = total_gap / float(max(1, matched))
    return out_entries, DistillStats(matched=matched, added=added, avg_score_gap=float(avg_gap))
