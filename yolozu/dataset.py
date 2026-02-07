from pathlib import Path


def load_yolo_dataset(images_dir, labels_dir):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(sorted(images_dir.glob(ext)))
    images = sorted(images)
    records = []
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
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
        records.append({"image": str(image_path), "labels": labels})
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
    records = load_yolo_dataset(images_dir, labels_dir)
    return {"images": records, "split": split}
