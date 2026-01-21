from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ValidationResult:
    warnings: list[str]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_bbox(bbox: Any, *, strict: bool, where: str) -> list[str]:
    warnings: list[str] = []
    if not isinstance(bbox, dict):
        raise ValueError(f"{where}: bbox must be an object")
    for key in ("cx", "cy", "w", "h"):
        if key not in bbox:
            raise ValueError(f"{where}: bbox missing '{key}'")
        if strict and not _is_number(bbox[key]):
            raise ValueError(f"{where}: bbox.{key} must be a number")
    return warnings


def _validate_detection(det: Any, *, strict: bool, where: str) -> list[str]:
    warnings: list[str] = []
    if not isinstance(det, dict):
        raise ValueError(f"{where}: detection must be an object")

    # Minimal keys needed for most evaluation flows.
    if "score" not in det:
        raise ValueError(f"{where}: detection missing 'score'")
    if strict and not _is_number(det["score"]):
        raise ValueError(f"{where}: detection.score must be a number")

    if "bbox" not in det:
        raise ValueError(f"{where}: detection missing 'bbox'")
    warnings.extend(_validate_bbox(det["bbox"], strict=strict, where=f"{where}.bbox"))

    if "class_id" in det:
        if strict and not isinstance(det["class_id"], int):
            raise ValueError(f"{where}: detection.class_id must be int")
    else:
        warnings.append(f"{where}: detection missing 'class_id' (ok for some flows)")

    # Optional fields (RTDETRPoseAdapter schema)
    if "rot6d" in det:
        rot = det["rot6d"]
        if strict:
            if not isinstance(rot, list) or len(rot) != 6 or not all(_is_number(v) for v in rot):
                raise ValueError(f"{where}: detection.rot6d must be list[6] of numbers")
    if "offsets" in det:
        off = det["offsets"]
        if strict:
            if not isinstance(off, list) or len(off) != 2 or not all(_is_number(v) for v in off):
                raise ValueError(f"{where}: detection.offsets must be list[2] of numbers")
    if "k_delta" in det:
        kd = det["k_delta"]
        if strict:
            if not isinstance(kd, list) or len(kd) != 4 or not all(_is_number(v) for v in kd):
                raise ValueError(f"{where}: detection.k_delta must be list[4] of numbers")

    return warnings


def normalize_predictions_json(data: Any) -> list[dict[str, Any]]:
    """Normalize supported prediction JSON shapes into a list of entries.

    Supported:
      1) [{"image": "...", "detections": [...]}, ...]
      2) {"predictions": [ ...same as 1... ], ...}
      3) {"/path.jpg": [...], "0001.jpg": [...]}  (image->detections mapping)
    """

    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]

    if isinstance(data, list):
        out: list[dict[str, Any]] = []
        for entry in data:
            if isinstance(entry, dict):
                out.append(entry)
        return out

    if isinstance(data, dict):
        out = []
        for image, detections in data.items():
            out.append({"image": str(image), "detections": _as_list(detections)})
        return out

    raise ValueError("Unsupported predictions JSON format")


def validate_predictions_entries(entries: Iterable[dict[str, Any]], *, strict: bool = False) -> ValidationResult:
    warnings: list[str] = []
    for idx, entry in enumerate(entries):
        where = f"predictions[{idx}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: entry must be an object")
        image = entry.get("image")
        if not image:
            raise ValueError(f"{where}: missing 'image'")
        dets = entry.get("detections", [])
        if dets is None:
            dets = []
        if not isinstance(dets, list):
            raise ValueError(f"{where}: 'detections' must be a list")
        for j, det in enumerate(dets):
            warnings.extend(_validate_detection(det, strict=strict, where=f"{where}.detections[{j}]"))
    return ValidationResult(warnings=warnings)


def load_predictions_entries(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    data = json.loads(path.read_text())
    return normalize_predictions_json(data)


def load_predictions_index(path: str | Path, *, add_basename_aliases: bool = True) -> dict[str, list[Any]]:
    """Load predictions into an index mapping image key -> detections list."""

    entries = load_predictions_entries(path)

    index: dict[str, list[Any]] = {}
    for entry in entries:
        image = entry.get("image")
        if not image:
            continue
        dets = entry.get("detections", [])
        if dets is None:
            dets = []
        index[str(image)] = dets if isinstance(dets, list) else _as_list(dets)

    if add_basename_aliases:
        for image, dets in list(index.items()):
            base = str(image).split("/")[-1]
            if base and base not in index:
                index[base] = dets

    return index
