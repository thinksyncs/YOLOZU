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


def _resolve_optional_value(value, root):
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict)):
        return value
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    return str(path)


def _first_key(data, keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _load_metadata(meta_path, root):
    if not meta_path.exists():
        return {}
    data = json.loads(meta_path.read_text())
    metadata = {}
    mask_value = _first_key(data, ("mask_path", "M_path", "M", "mask"))
    depth_value = _first_key(data, ("depth_path", "D_obj_path", "D_obj", "depth"))
    if mask_value is not None:
        resolved = _resolve_optional_value(mask_value, root)
        if isinstance(resolved, str):
            metadata["mask_path"] = resolved
        else:
            metadata["mask"] = resolved
    if depth_value is not None:
        resolved = _resolve_optional_value(depth_value, root)
        if isinstance(resolved, str):
            metadata["depth_path"] = resolved
        else:
            metadata["depth"] = resolved

    if "pose" in data:
        metadata["pose"] = data["pose"]
    r_gt = _first_key(data, ("R_gt", "R"))
    t_gt = _first_key(data, ("t_gt", "t"))
    if r_gt is not None:
        metadata["R_gt"] = r_gt
    if t_gt is not None:
        metadata["t_gt"] = t_gt
    if "pose" not in metadata and (r_gt is not None or t_gt is not None):
        metadata["pose"] = {"R": r_gt, "t": t_gt}

    if "intrinsics" in data:
        metadata["intrinsics"] = data["intrinsics"]
    k_gt = _first_key(data, ("K_gt", "K"))
    if k_gt is not None:
        metadata["K_gt"] = k_gt
        metadata["intrinsics"] = k_gt

    k_gt_prime = _first_key(data, ("K_gt_prime", "K_gt'"))
    if k_gt_prime is not None:
        metadata["K_gt_prime"] = k_gt_prime
        metadata["intrinsics_prime"] = k_gt_prime
    cad_value = data.get("cad_points") or data.get("cad_path") or data.get("cad_points_path")
    if cad_value is not None:
        metadata["cad_points"] = _resolve_optional_value(cad_value, root)
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
                "mask": metadata.get("mask"),
                "depth_path": metadata.get("depth_path"),
                "depth": metadata.get("depth"),
                "pose": metadata.get("pose"),
                "R_gt": metadata.get("R_gt"),
                "t_gt": metadata.get("t_gt"),
                "intrinsics": metadata.get("intrinsics"),
                "K_gt": metadata.get("K_gt"),
                "intrinsics_prime": metadata.get("intrinsics_prime"),
                "K_gt_prime": metadata.get("K_gt_prime"),
                "cad_points": metadata.get("cad_points"),
                "M": metadata.get("mask"),
                "D_obj": metadata.get("depth"),
            }
        )
    return records


def build_manifest(root, split="train2017"):
    return {"images": load_yolo_dataset(root, split=split)}
