from pathlib import Path


def load_yolo_dataset(images_dir, labels_dir):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    images = sorted(images_dir.glob("*.jpg"))
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


def build_manifest(dataset_root):
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images" / "train2017"
    labels_dir = dataset_root / "labels" / "train2017"
    records = load_yolo_dataset(images_dir, labels_dir)
    return {"images": records}
