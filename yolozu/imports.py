from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import TrainConfig
from .config import simple_yaml_load
from .coco_convert import build_category_map_from_coco


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
            return data or {}
        except Exception:
            return simple_yaml_load(text)
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        return json.loads(text)
    except Exception:
        return simple_yaml_load(text)


def import_coco_instances_dataset(
    *,
    instances_json: str | Path,
    images_dir: str | Path,
    split: str,
    output: str | Path,
    include_crowd: bool = False,
    force: bool = False,
) -> Path:
    """Create a read-only dataset wrapper for COCO instances JSON."""

    instances_path = Path(instances_json)
    if not instances_path.is_absolute():
        instances_path = (Path.cwd() / instances_path).resolve()

    images_dir_path = Path(images_dir)
    if not images_dir_path.is_absolute():
        images_dir_path = (Path.cwd() / images_dir_path).resolve()

    out_path = Path(output)
    if out_path.suffix.lower() != ".json":
        out_root = out_path
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "dataset.json"
    else:
        out_root = out_path.parent
        out_root.mkdir(parents=True, exist_ok=True)

    if not instances_path.exists():
        raise FileNotFoundError(f"--instances not found: {instances_path}")
    if not images_dir_path.exists():
        raise FileNotFoundError(f"--images-dir not found: {images_dir_path}")
    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")

    instances_doc = json.loads(instances_path.read_text(encoding="utf-8"))
    cat_map = build_category_map_from_coco(instances_doc)

    # Persist a stable classes.json mapping under labels/<split>/ for downstream tools.
    labels_dir = out_root / "labels" / str(split)
    labels_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "category_id_to_class_id": cat_map.category_id_to_class_id,
        "class_id_to_category_id": cat_map.class_id_to_category_id,
        "class_names": cat_map.class_names,
    }
    (labels_dir / "classes.json").write_text(json.dumps(mapping, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (labels_dir / "classes.txt").write_text("\n".join(cat_map.class_names) + "\n", encoding="utf-8")

    payload: dict[str, Any] = {
        "format": "coco_instances",
        "instances_json": str(instances_path),
        "images_dir": str(images_dir_path),
        "split": str(split),
        "include_crowd": bool(include_crowd),
        "source": {"from": "coco-instances"},
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def import_ultralytics_config(
    *,
    args_yaml: str | Path,
    output: str | Path,
    force: bool = False,
) -> Path:
    args_path = Path(args_yaml)
    if not args_path.is_absolute():
        args_path = (Path.cwd() / args_path).resolve()
    out_path = Path(output)
    if not args_path.exists():
        raise FileNotFoundError(f"--args not found: {args_path}")

    if out_path.suffix.lower() != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "train_config_import.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")

    cfg = _load_config(args_path)

    imgsz = cfg.get("imgsz")
    batch = cfg.get("batch")
    epochs = cfg.get("epochs")
    lr0 = cfg.get("lr0")
    wd = cfg.get("weight_decay")
    optimizer = cfg.get("optimizer")
    seed = cfg.get("seed")
    device = cfg.get("device")
    model = cfg.get("model")

    preprocess: dict[str, Any] = {}
    for key in ("imgsz", "rect", "single_cls", "multi_scale", "cache"):
        if key in cfg:
            preprocess[key] = cfg.get(key)

    train = TrainConfig(
        model=(str(model) if isinstance(model, str) and model.strip() else None),
        imgsz=(imgsz if isinstance(imgsz, (int, list)) else None),
        batch=(int(batch) if isinstance(batch, int) and not isinstance(batch, bool) else None),
        epochs=(int(epochs) if isinstance(epochs, int) and not isinstance(epochs, bool) else None),
        optimizer=(str(optimizer) if isinstance(optimizer, str) and optimizer.strip() else None),
        lr=(float(lr0) if isinstance(lr0, (int, float)) and not isinstance(lr0, bool) else None),
        weight_decay=(float(wd) if isinstance(wd, (int, float)) and not isinstance(wd, bool) else None),
        seed=(int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None),
        device=(str(device) if isinstance(device, str) and device.strip() else None),
        preprocess=preprocess or None,
        source={"from": "ultralytics", "args_yaml": str(args_path)},
    )

    out_path.write_text(json.dumps(train.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
