import json
from pathlib import Path


def _load_yolo_labels(label_path):
    labels = []
    if not label_path.exists():
        return labels
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
                "bbox": {
                    "cx": coords[0],
                    "cy": coords[1],
                    "w": coords[2],
                    "h": coords[3],
                },
            }
        )
    return labels


def _resolve_optional_path(value, root):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    return str(path)


def _load_metadata(meta_path, root):
    if not meta_path.exists():
        return {}
    data = json.loads(meta_path.read_text())
    metadata = {}
    mask_value = (
        data.get("mask_path")
        or data.get("M_path")
        or data.get("M")
        or data.get("mask")
    )
    depth_value = (
        data.get("depth_path")
        or data.get("D_obj_path")
        or data.get("D_obj")
        or data.get("depth")
    )
    if mask_value is not None:
        metadata["mask_path"] = _resolve_optional_path(mask_value, root)
    if depth_value is not None:
        metadata["depth_path"] = _resolve_optional_path(depth_value, root)

    if "pose" in data:
        metadata["pose"] = data["pose"]
    elif "R_gt" in data or "t_gt" in data:
        metadata["pose"] = {"R": data.get("R_gt"), "t": data.get("t_gt")}

    if "intrinsics" in data:
        metadata["intrinsics"] = data["intrinsics"]
    elif "K_gt" in data:
        metadata["intrinsics"] = data["K_gt"]
    elif "K" in data:
        metadata["intrinsics"] = data["K"]

    if "K_gt_prime" in data:
        metadata["intrinsics_prime"] = data["K_gt_prime"]
    if "K_gt'" in data:
        metadata["intrinsics_prime"] = data["K_gt'"]
    cad_value = data.get("cad_points") or data.get("cad_path") or data.get("cad_points_path")
    if cad_value is not None:
        metadata["cad_points"] = _resolve_optional_path(cad_value, root)
    return metadata


def load_yolo_dataset(root, split="train2017"):
    root = Path(root)
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    records = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        meta_path = labels_dir / f"{image_path.stem}.json"
        labels = _load_yolo_labels(label_path)
        metadata = _load_metadata(meta_path, root)
        records.append(
            {
                "image_path": str(image_path),
                "labels": labels,
                "mask_path": metadata.get("mask_path"),
                "depth_path": metadata.get("depth_path"),
                "pose": metadata.get("pose"),
                "intrinsics": metadata.get("intrinsics"),
                "intrinsics_prime": metadata.get("intrinsics_prime"),
                "cad_points": metadata.get("cad_points"),
            }
        )
    return records


def build_manifest(root, split="train2017"):
    return {"images": load_yolo_dataset(root, split=split)}
