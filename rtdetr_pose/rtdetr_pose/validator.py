from pathlib import Path


def _shape(value):
    if hasattr(value, "shape"):
        try:
            return tuple(int(dim) for dim in value.shape)
        except TypeError:
            return None
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return (len(value), len(value[0]))
        return (len(value),)
    return None


def _is_matrix(value, size):
    shape = _shape(value)
    if shape is None:
        return False
    return len(shape) == 2 and shape[0] == size and shape[1] == size


def _is_vector(value, size):
    shape = _shape(value)
    if shape is None:
        return False
    return len(shape) == 1 and shape[0] == size


def _validate_optional_path_or_array(value, name):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise ValueError(f"{name} missing: {path}")
        return ("path", None)
    shape = _shape(value)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be a path or 2D array")
    return ("array", shape)


def _validate_pose(value):
    if value is None:
        return
    if isinstance(value, dict):
        r = value.get("R") or value.get("R_gt")
        t = value.get("t") or value.get("t_gt")
        if r is None or t is None:
            raise ValueError("pose must include R and t")
        if not _is_matrix(r, 3):
            raise ValueError("pose R must be 3x3")
        if not _is_vector(t, 3):
            raise ValueError("pose t must be length 3")
        return
    if _is_matrix(value, 4):
        return
    raise ValueError("pose must be a dict with R/t or a 4x4 matrix")


def _validate_intrinsics(value):
    if value is None:
        return
    if isinstance(value, dict):
        if all(k in value for k in ("fx", "fy", "cx", "cy")):
            return
        if "K" in value and _is_matrix(value["K"], 3):
            return
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            return
    if _is_matrix(value, 3):
        return
    raise ValueError("intrinsics must be fx/fy/cx/cy or 3x3 matrix")


REQUIRED_KEYS = {
    "image_path",
    "labels",
    "mask_path",
    "depth_path",
    "pose",
    "intrinsics",
}


def validate_sample(sample, strict=False):
    missing = REQUIRED_KEYS - set(sample.keys())
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    image_path = Path(sample["image_path"])
    if not image_path.exists():
        raise ValueError(f"image missing: {image_path}")
    if not isinstance(sample["labels"], list):
        raise ValueError("labels must be list")
    for label in sample["labels"]:
        if "class_id" not in label or "bbox" not in label:
            raise ValueError("label missing class_id/bbox")
        bbox = label["bbox"]
        for key in ("cx", "cy", "w", "h"):
            value = bbox.get(key)
            if value is None or not (0.0 <= value <= 1.0):
                raise ValueError(f"bbox {key} out of range: {value}")
    mask_info = _validate_optional_path_or_array(sample.get("mask_path"), "mask_path")
    depth_info = _validate_optional_path_or_array(sample.get("depth_path"), "depth_path")
    if mask_info and depth_info:
        if mask_info[0] == "array" and depth_info[0] == "array":
            if mask_info[1] != depth_info[1]:
                raise ValueError("mask/depth shapes must match")
    _validate_pose(sample.get("pose"))
    _validate_intrinsics(sample.get("intrinsics"))
    if strict:
        if sample["mask_path"] is None:
            raise ValueError("mask_path required in strict mode")
        if sample["depth_path"] is None:
            raise ValueError("depth_path required in strict mode")
        if sample["pose"] is None:
            raise ValueError("pose required in strict mode")
        if sample["intrinsics"] is None:
            raise ValueError("intrinsics required in strict mode")


def validate_manifest(manifest, strict=False):
    images = manifest.get("images", [])
    if not images:
        raise ValueError("manifest has no images")
    for sample in images:
        validate_sample(sample, strict=strict)
