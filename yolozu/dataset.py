from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _first_key(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _resolve_optional_path(value: Any, root: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return list(value)
        if isinstance(value[0], (str, Path)):
            out = []
            for item in value:
                if item is None:
                    out.append(None)
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = root / path
                out.append(str(path))
            return out
        return value
    if isinstance(value, str) or isinstance(value, Path):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return str(path)
    return value


def _load_sidecar_metadata(meta_path: Path, root: Path) -> dict[str, Any]:
    """Load optional per-image metadata from labels/<split>/<image>.json.

    This loader is intentionally lightweight: it resolves relative paths but does
    not decode masks/depth arrays. The resulting keys are merged into the record.
    """

    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text())
    except Exception:
        return {}

    out: dict[str, Any] = {}

    # Optional image size hints (used for pixel conversion).
    image_hw = data.get("image_hw") or data.get("hw")
    image_size = data.get("image_size")
    if image_hw is not None:
        out["image_hw"] = image_hw
    if image_size is not None:
        out["image_size"] = image_size

    # Pose / intrinsics.
    if "pose" in data:
        out["pose"] = data.get("pose")
    r_gt = _first_key(data, ("R_gt", "R"))
    t_gt = _first_key(data, ("t_gt", "t"))
    if r_gt is not None:
        out["R_gt"] = r_gt
    if t_gt is not None:
        out["t_gt"] = t_gt
    if "pose" not in out and (r_gt is not None or t_gt is not None):
        out["pose"] = {"R": r_gt, "t": t_gt}

    if "intrinsics" in data:
        out["intrinsics"] = data.get("intrinsics")
    k_gt = _first_key(data, ("K_gt", "K"))
    if k_gt is not None:
        out["K_gt"] = k_gt
        out["intrinsics"] = k_gt

    # Optional paths (kept as paths/inline values).
    mask_value = _first_key(data, ("mask_path", "M_path", "M", "mask"))
    depth_value = _first_key(data, ("depth_path", "D_obj_path", "D_obj", "depth"))
    if mask_value is not None:
        resolved = _resolve_optional_path(mask_value, root)
        out["mask_path"] = resolved
        out["mask"] = resolved
    if depth_value is not None:
        resolved = _resolve_optional_path(depth_value, root)
        out["depth_path"] = resolved
        out["depth"] = resolved
        out["D_obj"] = resolved

    return out


def load_yolo_dataset(images_dir, labels_dir, *, dataset_root: Path | None = None):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    dataset_root = Path(dataset_root) if dataset_root is not None else labels_dir.parent.parent
    images = sorted(images_dir.glob("*.jpg"))
    records = []
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        meta_path = labels_dir / f"{image_path.stem}.json"
        labels = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 5:
                    raise ValueError(f"invalid label line: {line}")
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
                labels.append(
                    {
                        "class_id": class_id,
                        "cx": coords[0],
                        "cy": coords[1],
                        "w": coords[2],
                        "h": coords[3],
                    }
                )
        record = {"image": str(image_path), "labels": labels}
        record.update(_load_sidecar_metadata(meta_path, dataset_root))
        records.append(record)
    return records


def _pick_split(dataset_root: Path, split: str | None) -> str:
    if split:
        return split
    for candidate in ("val2017", "train2017"):
        if (dataset_root / "images" / candidate).exists():
            return candidate
    return "train2017"


def build_manifest(dataset_root, *, split: str | None = None):
    dataset_root = Path(dataset_root)
    split = _pick_split(dataset_root, split)
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    records = load_yolo_dataset(images_dir, labels_dir, dataset_root=dataset_root)
    return {"images": records, "split": split}
