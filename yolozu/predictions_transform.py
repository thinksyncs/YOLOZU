from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TransformResult:
    entries: list[dict[str, Any]]
    warnings: list[str]


def load_classes_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _to_int_key_map(mapping: dict[Any, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for k, v in mapping.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def build_category_id_to_class_id_map(classes_json: dict[str, Any]) -> dict[int, int]:
    raw = classes_json.get("category_id_to_class_id")
    if not isinstance(raw, dict):
        raise ValueError("classes.json missing category_id_to_class_id")
    return _to_int_key_map(raw)


def normalize_class_ids(
    entries: Iterable[dict[str, Any]],
    *,
    classes_json: dict[str, Any] | None = None,
    assume_class_id_is_category_id: bool = False,
) -> TransformResult:
    """Normalize detections to use contiguous `class_id` (0..N-1).

    Supported normalization:
    - If a detection has `category_id` and is missing `class_id`, map it.
    - If assume_class_id_is_category_id=True, treat `class_id` as a COCO category id and map it.
    """

    warnings: list[str] = []
    cat_to_cls: dict[int, int] | None = None
    if classes_json is not None:
        cat_to_cls = build_category_id_to_class_id_map(classes_json)

    out_entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        dets = new_entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []

        new_dets = []
        for j, det in enumerate(dets):
            if not isinstance(det, dict):
                continue
            new_det = dict(det)

            if assume_class_id_is_category_id and "class_id" in new_det and cat_to_cls is not None:
                try:
                    cat_id = int(new_det["class_id"])
                    if cat_id in cat_to_cls:
                        new_det["class_id"] = int(cat_to_cls[cat_id])
                    else:
                        warnings.append(f"predictions[{idx}].detections[{j}]: unknown category_id {cat_id}")
                except Exception:
                    warnings.append(f"predictions[{idx}].detections[{j}]: invalid class_id")

            if "class_id" not in new_det and "category_id" in new_det and cat_to_cls is not None:
                try:
                    cat_id = int(new_det["category_id"])
                    if cat_id in cat_to_cls:
                        new_det["class_id"] = int(cat_to_cls[cat_id])
                    else:
                        warnings.append(f"predictions[{idx}].detections[{j}]: unknown category_id {cat_id}")
                except Exception:
                    warnings.append(f"predictions[{idx}].detections[{j}]: invalid category_id")

            new_dets.append(new_det)

        new_entry["detections"] = new_dets
        out_entries.append(new_entry)

    return TransformResult(entries=out_entries, warnings=warnings)

