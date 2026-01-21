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
        if not value:
            return list(value)
        # If this is a list of paths (strings), resolve each element.
        if isinstance(value[0], (str, Path)):
            resolved = []
            for item in value:
                if item is None:
                    resolved.append(None)
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = Path(root) / path
                resolved.append(str(path))
            return resolved
        # Otherwise keep as-is (likely an inline array).
        return value
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    return str(path)


def _resolve_optional_value(value, root):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return list(value)
        # List of relative/absolute paths.
        if isinstance(value[0], (str, Path)):
            resolved = []
            for item in value:
                if item is None:
                    resolved.append(None)
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = Path(root) / path
                resolved.append(str(path))
            return resolved
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
        # Keep using the legacy key name `mask_path` because the validator treats it as
        # "path OR 2D array". This preserves strict-mode behavior for inline masks.
        metadata["mask_path"] = resolved
        metadata["mask"] = resolved
    if depth_value is not None:
        resolved = _resolve_optional_value(depth_value, root)
        metadata["depth_path"] = resolved
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

        k_raw = metadata.get("K_gt")
        if k_raw is None:
            k_raw = metadata.get("intrinsics")
        k_3x3 = _k_to_3x3(k_raw)

        k_prime_raw = metadata.get("K_gt_prime")
        if k_prime_raw is None:
            k_prime_raw = metadata.get("intrinsics_prime")
        k_prime_3x3 = _k_to_3x3(k_prime_raw)

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
                "K_gt": k_3x3 if k_3x3 is not None else metadata.get("K_gt"),
                "intrinsics_prime": metadata.get("intrinsics_prime"),
                "K_gt_prime": k_prime_3x3 if k_prime_3x3 is not None else metadata.get("K_gt_prime"),
                "cad_points": metadata.get("cad_points"),
                "M": metadata.get("mask"),
                "D_obj": metadata.get("depth"),
            }
        )
    return records


def _availability_stats(records):
    stats = {
        "total": len(records),
        "mask": 0,
        "depth": 0,
        "pose": 0,
        "intrinsics": 0,
        "cad_points": 0,
    }
    for rec in records:
        if rec.get("mask_path") is not None:
            stats["mask"] += 1
        if rec.get("depth_path") is not None:
            stats["depth"] += 1
        if rec.get("pose") is not None or rec.get("R_gt") is not None or rec.get("t_gt") is not None:
            stats["pose"] += 1
        if rec.get("intrinsics") is not None or rec.get("K_gt") is not None:
            stats["intrinsics"] += 1
        if rec.get("cad_points") is not None:
            stats["cad_points"] += 1
    return stats


def build_manifest(root, split="train2017"):
    images = load_yolo_dataset(root, split=split)
    return {"images": images, "stats": _availability_stats(images)}


def _k_to_3x3(k):
    if k is None:
        return None
    # K may be a 3x3 nested list, or (fx, fy, cx, cy)
    if isinstance(k, (list, tuple)) and len(k) == 4 and not isinstance(k[0], (list, tuple)):
        fx, fy, cx, cy = [float(v) for v in k]
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    if isinstance(k, (list, tuple)) and len(k) == 3 and isinstance(k[0], (list, tuple)):
        return [[float(v) for v in row] for row in k]
    if isinstance(k, dict) and {"fx", "fy", "cx", "cy"}.issubset(k.keys()):
        fx = float(k["fx"])
        fy = float(k["fy"])
        cx = float(k["cx"])
        cy = float(k["cy"])
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    return None


def _expand_to_instances(value, num_instances, *, default=None):
    if value is None:
        return [default for _ in range(int(num_instances))]
    if isinstance(value, (list, tuple)) and len(value) == int(num_instances):
        return list(value)
    return [value for _ in range(int(num_instances))]


def extract_pose_intrinsics_targets(record, num_instances: int):
    """Extract and normalize optional pose/intrinsics targets.

    Returns a dict of python lists suitable for conversion to torch tensors:
    - K_gt: 3x3 nested list or None
    - t_gt: list length num_instances, each item is [x,y,z] or None
    - R_gt: list length num_instances, each item is 3x3 nested list or None
    - offsets_gt: list length num_instances, each item is [du,dv] or None
    """

    pose = record.get("pose") or {}
    r_gt = record.get("R_gt")
    t_gt = record.get("t_gt")
    if r_gt is None and isinstance(pose, dict):
        r_gt = pose.get("R")
    if t_gt is None and isinstance(pose, dict):
        t_gt = pose.get("t")

    k_raw = record.get("K_gt")
    if k_raw is None:
        k_raw = record.get("intrinsics")
    k_gt = _k_to_3x3(k_raw)

    # R and t may be per-image or per-instance.
    r_list = _expand_to_instances(r_gt, num_instances, default=None)
    t_list = _expand_to_instances(t_gt, num_instances, default=None)

    def _norm_R(r):
        if r is None:
            return None
        if isinstance(r, (list, tuple)) and len(r) == 3 and isinstance(r[0], (list, tuple)):
            return [[float(v) for v in row] for row in r]
        return None

    def _norm_t(t):
        if t is None:
            return None
        if isinstance(t, (list, tuple)) and len(t) == 3 and not isinstance(t[0], (list, tuple)):
            return [float(v) for v in t]
        return None

    r_list = [_norm_R(r) for r in r_list]
    t_list = [_norm_t(t) for t in t_list]

    offsets_raw = record.get("offsets_gt")
    if offsets_raw is None:
        offsets_raw = record.get("offsets")
    off_list = _expand_to_instances(offsets_raw, num_instances, default=None)

    def _norm_off(off):
        if off is None:
            return None
        if isinstance(off, (list, tuple)) and len(off) == 2:
            return [float(off[0]), float(off[1])]
        return None

    off_list = [_norm_off(o) for o in off_list]

    return {
        "K_gt": k_gt,
        "t_gt": t_list,
        "R_gt": r_list,
        "offsets_gt": off_list,
    }


def extract_full_gt_targets(record, num_instances: int):
    """Extract and normalize full GT fields per spec.

    Returns canonical python lists suitable for conversion to torch tensors
    (where applicable) without eager decoding:

    - M: list length num_instances, each item is a 2D array OR a path string OR None
    - D_obj: list length num_instances, each item is a 2D array OR a path string OR None
    - R_gt/t_gt/K_gt/offsets_gt: see extract_pose_intrinsics_targets
    - *_mask: list[bool] indicating availability per instance
    """

    base = extract_pose_intrinsics_targets(record, num_instances=num_instances)

    m_raw = record.get("M")
    if m_raw is None:
        m_raw = record.get("mask")
    if m_raw is None:
        m_raw = record.get("mask_path")

    d_raw = record.get("D_obj")
    if d_raw is None:
        d_raw = record.get("depth")
    if d_raw is None:
        d_raw = record.get("depth_path")

    m_list = _expand_to_instances(m_raw, num_instances, default=None)
    d_list = _expand_to_instances(d_raw, num_instances, default=None)

    def _norm_m(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return value
        return None

    def _norm_d(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return value
        return None

    m_list = [_norm_m(v) for v in m_list]
    d_list = [_norm_d(v) for v in d_list]
    m_mask = [v is not None for v in m_list]
    d_mask = [v is not None for v in d_list]

    return {
        **base,
        "M": m_list,
        "D_obj": d_list,
        "M_mask": m_mask,
        "D_obj_mask": d_mask,
    }


def depth_at_bbox_center(d_obj, bbox, mask=None):
    """Sample depth (z) at bbox center from an object-only depth map.

    d_obj: 2D array-like (nested lists/tuples or ndarray)
    bbox: dict with cx/cy or sequence [cx,cy,w,h] in normalized coords
    mask: optional 2D mask (same spatial size); if provided and center is masked out,
          returns None.
    """

    if d_obj is None:
        return None

    # Extract bbox center in normalized coords.
    if isinstance(bbox, dict):
        cx = float(bbox.get("cx", 0.0))
        cy = float(bbox.get("cy", 0.0))
    else:
        cx = float(bbox[0])
        cy = float(bbox[1])

    # Determine H,W.
    if hasattr(d_obj, "shape"):
        h = int(d_obj.shape[0])
        w = int(d_obj.shape[1])
    else:
        if not isinstance(d_obj, (list, tuple)) or not d_obj:
            return None
        h = len(d_obj)
        w = len(d_obj[0]) if isinstance(d_obj[0], (list, tuple)) else 0
    if h <= 0 or w <= 0:
        return None

    u = int(round(cx * (w - 1)))
    v = int(round(cy * (h - 1)))
    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))

    if mask is not None:
        try:
            m_val = mask[v][u] if not hasattr(mask, "shape") else mask[v, u]
        except Exception:
            m_val = None
        if m_val is not None and float(m_val) <= 0.0:
            return None

    try:
        z = d_obj[v][u] if not hasattr(d_obj, "shape") else d_obj[v, u]
    except Exception:
        return None

    try:
        zf = float(z)
    except Exception:
        return None
    if not (zf >= 0.0):
        return None
    return zf
