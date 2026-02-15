from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .boxes import iou_cxcywh_norm_dict
from .geometry import corrected_intrinsics, recover_translation
from .image_keys import add_image_aliases, image_basename, lookup_image_alias
from .intrinsics import parse_intrinsics
from .math3d import geodesic_distance


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except Exception:
            return None
    if hasattr(value, "tolist"):
        try:
            return _as_float_list(value.tolist())
        except Exception:
            return None
    return None


def _as_matrix_3x3(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        if len(value) == 3 and isinstance(value[0], (list, tuple)) and len(value[0]) == 3:
            try:
                return [[float(x) for x in row] for row in value]  # type: ignore[misc]
            except Exception:
                return None
        if len(value) == 9 and not isinstance(value[0], (list, tuple, dict)):
            try:
                flat = [float(v) for v in value]
            except Exception:
                return None
            return [
                [flat[0], flat[1], flat[2]],
                [flat[3], flat[4], flat[5]],
                [flat[6], flat[7], flat[8]],
            ]
    return None


def _expand_to_instances(value: Any, n: int) -> list[Any | None]:
    if value is None:
        return [None] * n
    if isinstance(value, (list, tuple)):
        # Per-instance list (heuristic: length==n and first element is not scalar).
        if len(value) == n and (n == 0 or not isinstance(value[0], (int, float, str))):
            return list(value)
        return [value] * n
    return [value] * n


def _extract_r_gt(record: dict[str, Any], n: int) -> list[list[list[float]] | None]:
    pose = record.get("pose") if isinstance(record.get("pose"), dict) else {}
    r_raw = record.get("R_gt")
    if r_raw is None and isinstance(pose, dict):
        r_raw = pose.get("R")
    r_list = _expand_to_instances(r_raw, n)
    out: list[list[list[float]] | None] = []
    for item in r_list:
        out.append(_as_matrix_3x3(item))
    return out


def _extract_t_gt(record: dict[str, Any], n: int) -> list[list[float] | None]:
    pose = record.get("pose") if isinstance(record.get("pose"), dict) else {}
    t_raw = record.get("t_gt")
    if t_raw is None and isinstance(pose, dict):
        t_raw = pose.get("t")
    t_list = _expand_to_instances(t_raw, n)
    out: list[list[float] | None] = []
    for item in t_list:
        vals = _as_float_list(item)
        if vals is None or len(vals) != 3:
            out.append(None)
        else:
            out.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return out


def _rot6d_to_matrix(rot6d: Any) -> list[list[float]] | None:
    vals = _as_float_list(rot6d)
    if vals is None or len(vals) != 6:
        return None
    a1 = vals[0:3]
    a2 = vals[3:6]

    def _norm(v: list[float]) -> list[float] | None:
        n = float((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5)
        if n <= 0.0:
            return None
        return [v[0] / n, v[1] / n, v[2] / n]

    b1 = _norm(a1)
    if b1 is None:
        return None
    dot = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2]
    a2o = [a2[0] - dot * b1[0], a2[1] - dot * b1[1], a2[2] - dot * b1[2]]
    b2 = _norm(a2o)
    if b2 is None:
        return None
    b3 = [
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    ]
    return [b1, b2, b3]


def _parse_image_wh(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, dict):
        try:
            return float(value.get("width")), float(value.get("height"))
        except Exception:
            return (None, None)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return (None, None)
    return (None, None)


def _bbox_center_px(bbox: dict[str, Any], *, image_wh: tuple[float | None, float | None]) -> tuple[float, float] | None:
    if not isinstance(bbox, dict):
        return None
    try:
        cx = float(bbox["cx"])
        cy = float(bbox["cy"])
    except Exception:
        return None
    w, h = image_wh
    if w is None or h is None or w <= 0 or h <= 0:
        return None
    return (cx * w, cy * h)


def _bbox_iou_cxcywh_norm(a: dict[str, Any], b: dict[str, Any]) -> float:
    return iou_cxcywh_norm_dict(a, b)


def _match_dets_to_gts(
    dets: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[tuple[float, int, int]]:
    if not dets or not gts:
        return []

    pairs: list[tuple[float, int, int]] = []
    for det_idx, det in enumerate(dets):
        bbox = det.get("bbox")
        if not isinstance(bbox, dict):
            continue
        det_cls = det.get("class_id")
        for gt_idx, gt in enumerate(gts):
            if det_cls is not None and det_cls != gt.get("class_id"):
                continue
            gt_bbox = gt.get("bbox")
            if not isinstance(gt_bbox, dict):
                continue
            try:
                iou = _bbox_iou_cxcywh_norm(bbox, gt_bbox)
            except Exception:
                continue
            if iou >= float(iou_threshold):
                pairs.append((float(iou), int(det_idx), int(gt_idx)))

    pairs.sort(reverse=True, key=lambda x: x[0])
    used_det: set[int] = set()
    used_gt: set[int] = set()
    out: list[tuple[float, int, int]] = []
    for iou, det_idx, gt_idx in pairs:
        if det_idx in used_det or gt_idx in used_gt:
            continue
        used_det.add(det_idx)
        used_gt.add(gt_idx)
        out.append((float(iou), int(det_idx), int(gt_idx)))
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    vs = sorted(float(v) for v in values)
    m = len(vs) // 2
    if len(vs) % 2 == 1:
        return float(vs[m])
    return float(0.5 * (vs[m - 1] + vs[m]))


def _success_rate(values: list[float], *, threshold: float) -> float | None:
    if not values:
        return None
    ok = sum(1 for v in values if float(v) <= float(threshold))
    return float(ok) / float(len(values))


@dataclass(frozen=True)
class PoseEvalResult:
    metrics: dict[str, float | None]
    counts: dict[str, int]
    per_image: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def evaluate_pose(
    records: list[dict[str, Any]],
    predictions_entries: Iterable[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
    min_score: float = 0.0,
    success_rot_deg: float = 15.0,
    success_trans: float = 0.1,
    keep_per_image: int = 0,
) -> PoseEvalResult:
    """Evaluate pose/depth errors after matching detections to GT by bbox IoU.

    - Matches are class-consistent and greedy by IoU.
    - Rotation error: geodesic distance between rotation matrices, reported in degrees.
    - Translation errors: L2 distance and |z_pred - z_gt| when both are available.
    """

    pred_index: dict[str, dict[str, Any]] = {}
    for entry in predictions_entries:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if not isinstance(image, str) or not image:
            continue
        add_image_aliases(pred_index, str(image), entry)

    total_gt = 0
    total_pred = 0
    total_matches = 0

    ious: list[float] = []
    rot_deg: list[float] = []
    trans_l2: list[float] = []
    depth_abs: list[float] = []
    pose_pairs = 0
    pose_pairs_ok = 0

    per_image: list[dict[str, Any]] = []
    warnings: list[str] = []

    for record in records:
        image = record.get("image")
        if not isinstance(image, str) or not image:
            continue
        labels = record.get("labels") or []
        if not isinstance(labels, list):
            labels = []

        gts: list[dict[str, Any]] = []
        for lab in labels:
            if not isinstance(lab, dict):
                continue
            try:
                cid = int(lab.get("class_id", 0))
                bbox = {
                    "cx": float(lab["cx"]),
                    "cy": float(lab["cy"]),
                    "w": float(lab["w"]),
                    "h": float(lab["h"]),
                }
            except Exception:
                continue
            gts.append({"class_id": cid, "bbox": bbox})

        n_gt = int(len(gts))
        total_gt += n_gt

        r_gt_list = _extract_r_gt(record, n_gt)
        t_gt_list = _extract_t_gt(record, n_gt)

        entry = lookup_image_alias(pred_index, image)
        dets_raw = entry.get("detections", []) if isinstance(entry, dict) else []
        dets_list = dets_raw if isinstance(dets_raw, list) else []

        dets: list[dict[str, Any]] = []
        image_wh = _parse_image_wh(entry.get("image_size")) if isinstance(entry, dict) else (None, None)
        intr = parse_intrinsics(entry.get("intrinsics")) if isinstance(entry, dict) else None

        for det in dets_list:
            if not isinstance(det, dict):
                continue
            try:
                score = float(det.get("score", 0.0))
            except Exception:
                score = 0.0
            if score < float(min_score):
                continue
            if not isinstance(det.get("bbox"), dict):
                continue
            dets.append(det)

        total_pred += int(len(dets))

        matches = _match_dets_to_gts(dets, gts, iou_threshold=float(iou_threshold))
        total_matches += int(len(matches))

        img_rot: list[float] = []
        img_trans: list[float] = []
        img_depth: list[float] = []
        img_pose_pairs = 0
        img_pose_ok = 0
        for iou, det_idx, gt_idx in matches:
            ious.append(float(iou))
            det = dets[int(det_idx)]
            r_gt = r_gt_list[int(gt_idx)] if 0 <= int(gt_idx) < len(r_gt_list) else None
            t_gt = t_gt_list[int(gt_idx)] if 0 <= int(gt_idx) < len(t_gt_list) else None

            match_rot_deg = None
            match_trans_l2 = None

            r_pred = None
            if "R" in det:
                r_pred = _as_matrix_3x3(det.get("R"))
            if r_pred is None and "rot6d" in det:
                r_pred = _rot6d_to_matrix(det.get("rot6d"))

            if r_pred is not None and r_gt is not None:
                try:
                    geo = geodesic_distance(r_pred, r_gt)
                    deg = float(geo) * 180.0 / math.pi
                    rot_deg.append(float(deg))
                    img_rot.append(float(deg))
                    match_rot_deg = float(deg)
                except Exception:
                    warnings.append("rotation_eval_failed")

            z_pred = None
            if "t_xyz" in det:
                t_pred_vals = _as_float_list(det.get("t_xyz"))
                if t_pred_vals is not None and len(t_pred_vals) == 3:
                    z_pred = float(t_pred_vals[2])
            if z_pred is None and "log_z" in det:
                try:
                    z_pred = float(math.exp(float(det.get("log_z"))))
                except Exception:
                    z_pred = None
            if z_pred is None and "z" in det:
                try:
                    z_pred = float(det.get("z"))
                except Exception:
                    z_pred = None

            t_pred = None
            t_pred_vals = _as_float_list(det.get("t_xyz")) if isinstance(det.get("t_xyz"), (list, tuple)) else None
            if t_pred_vals is not None and len(t_pred_vals) == 3:
                t_pred = [float(t_pred_vals[0]), float(t_pred_vals[1]), float(t_pred_vals[2])]
            elif z_pred is not None and intr is not None:
                k = (float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"]))
                k_delta = det.get("k_delta")
                if not isinstance(k_delta, (list, tuple)) or len(k_delta) != 4:
                    k_delta = None
                k_prime = corrected_intrinsics(k, tuple(float(v) for v in k_delta)) if k_delta is not None else k
                offsets = det.get("offsets") or [0.0, 0.0]
                if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
                    offsets = [0.0, 0.0]
                center = _bbox_center_px(det.get("bbox"), image_wh=image_wh)
                if center is not None:
                    try:
                        t_xyz = recover_translation(center, (float(offsets[0]), float(offsets[1])), float(z_pred), k_prime)
                        t_pred = [float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2])]
                    except Exception:
                        t_pred = None

            if t_gt is not None and z_pred is not None:
                try:
                    dz = abs(float(z_pred) - float(t_gt[2]))
                    depth_abs.append(float(dz))
                    img_depth.append(float(dz))
                except Exception:
                    warnings.append("depth_eval_failed")

            if t_gt is not None and t_pred is not None:
                try:
                    dx = float(t_pred[0]) - float(t_gt[0])
                    dy = float(t_pred[1]) - float(t_gt[1])
                    dz = float(t_pred[2]) - float(t_gt[2])
                    d = float((dx * dx + dy * dy + dz * dz) ** 0.5)
                    trans_l2.append(float(d))
                    img_trans.append(float(d))
                    match_trans_l2 = float(d)
                except Exception:
                    warnings.append("translation_eval_failed")

            if match_rot_deg is not None and match_trans_l2 is not None:
                pose_pairs += 1
                img_pose_pairs += 1
                if float(match_rot_deg) <= float(success_rot_deg) and float(match_trans_l2) <= float(success_trans):
                    pose_pairs_ok += 1
                    img_pose_ok += 1

        if keep_per_image > 0:
            if len(per_image) < int(keep_per_image):
                per_image.append(
                    {
                        "image": image_basename(image) or image,
                        "gt": int(n_gt),
                        "pred": int(len(dets)),
                        "matches": int(len(matches)),
                        "match_rate": (float(len(matches)) / float(n_gt) if n_gt > 0 else None),
                        "rot_deg_mean": _mean(img_rot),
                        "trans_l2_mean": _mean(img_trans),
                        "depth_abs_mean": _mean(img_depth),
                        "iou_mean": _mean([float(m[0]) for m in matches]),
                        "pose_success": (float(img_pose_ok) / float(img_pose_pairs) if img_pose_pairs > 0 else None),
                    }
                )

    metrics: dict[str, float | None] = {
        "match_rate": (float(total_matches) / float(total_gt) if total_gt > 0 else None),
        "iou_mean": _mean(ious),
        "rot_deg_mean": _mean(rot_deg),
        "rot_deg_median": _median(rot_deg),
        "trans_l2_mean": _mean(trans_l2),
        "trans_l2_median": _median(trans_l2),
        "depth_abs_mean": _mean(depth_abs),
        "depth_abs_median": _median(depth_abs),
        "rot_success": _success_rate(rot_deg, threshold=float(success_rot_deg)),
        "trans_success": _success_rate(trans_l2, threshold=float(success_trans)),
        "pose_success": (float(pose_pairs_ok) / float(pose_pairs) if pose_pairs > 0 else None),
    }

    counts = {
        "gt_instances": int(total_gt),
        "pred_instances": int(total_pred),
        "matches": int(total_matches),
        "rot_measured": int(len(rot_deg)),
        "trans_measured": int(len(trans_l2)),
        "depth_measured": int(len(depth_abs)),
        "pose_measured": int(pose_pairs),
    }

    return PoseEvalResult(metrics=metrics, counts=counts, per_image=per_image, warnings=warnings)
