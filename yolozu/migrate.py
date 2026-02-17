from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import simple_yaml_load


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


def _normalize_task(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    task = value.strip().lower()
    if not task:
        return None
    if task in ("detect", "det"):
        return "detect"
    if task in ("segment", "seg"):
        return "segment"
    if task in ("pose", "keypoints"):
        return "pose"
    return None


def _as_path(value: Any, *, base: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(value.strip())
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def write_dataset_wrapper(
    output: str | Path,
    *,
    images_dir: str | Path,
    labels_dir: str | Path,
    split: str,
    label_format: str | None = None,
    source: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    out_path = Path(output)
    if out_path.suffix.lower() != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "dataset.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")

    payload: dict[str, Any] = {
        "images_dir": str(Path(images_dir)),
        "labels_dir": str(Path(labels_dir)),
        "split": str(split),
    }
    if label_format is not None:
        payload["label_format"] = str(label_format)
    if source is not None:
        payload["source"] = dict(source)

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def migrate_ultralytics_dataset_wrapper(
    *,
    data_yaml: str | Path | None,
    args_yaml: str | Path | None = None,
    split: str | None = None,
    task: str | None = None,
    output: str | Path,
    force: bool = False,
) -> Path:
    from .dataset import _resolve_ultralytics_data_yaml

    args_path = Path(args_yaml) if args_yaml is not None else None
    data_path = Path(data_yaml) if data_yaml is not None else None

    args_cfg: dict[str, Any] | None = None
    if args_path is not None:
        if not args_path.exists():
            raise FileNotFoundError(f"--args not found: {args_path}")
        args_cfg = _load_config(args_path)

    if data_path is None and args_cfg is not None:
        inferred = _as_path(args_cfg.get("data"), base=args_path.parent)
        if inferred is not None:
            data_path = inferred

    if data_path is None:
        raise ValueError("--data is required (or provide --args with a usable data: path)")
    if not data_path.exists():
        raise FileNotFoundError(f"--data not found: {data_path}")

    resolved = _resolve_ultralytics_data_yaml(Path(data_path), split)
    if resolved is None:
        raise ValueError(f"failed to resolve Ultralytics data.yaml: {data_path}")

    images_dir, labels_dir, split_effective, inferred_root, yaml_label_format = resolved

    task_norm = _normalize_task(task)
    args_task = _normalize_task(args_cfg.get("task")) if args_cfg is not None else None
    effective_task = task_norm or args_task

    label_format: str | None = None
    if effective_task == "segment":
        label_format = "segment"
    elif yaml_label_format is not None:
        label_format = str(yaml_label_format)

    source: dict[str, Any] = {"from": "ultralytics", "data_yaml": str(data_path)}
    if args_path is not None:
        source["args_yaml"] = str(args_path)
    if effective_task is not None:
        source["task"] = effective_task
    if inferred_root is not None:
        source["dataset_root"] = str(inferred_root)

    return write_dataset_wrapper(
        output,
        images_dir=images_dir,
        labels_dir=labels_dir,
        split=split_effective,
        label_format=label_format,
        source=source,
        force=force,
    )


def _maybe_relative(path: Path | None, *, root: Path, path_type: str) -> str | None:
    if path is None:
        return None
    if path_type == "absolute":
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def write_seg_dataset_descriptor(
    output: str | Path,
    *,
    dataset: str,
    split: str,
    ignore_index: int,
    samples: Iterable[dict[str, Any]],
    classes: list[str] | None,
    path_type: str = "absolute",
    mode: str = "manifest",
    source: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")

    payload: dict[str, Any] = {
        "dataset": str(dataset),
        "task": "semantic_segmentation",
        "split": str(split),
        "mode": str(mode),
        "path_type": str(path_type),
        "ignore_index": int(ignore_index),
        "samples": list(samples),
    }
    if classes is not None:
        payload["classes"] = list(classes)
    if source is not None:
        payload["source"] = dict(source)

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def migrate_seg_dataset_descriptor(
    *,
    from_format: str,
    root: str | Path,
    split: str,
    output: str | Path,
    path_type: str = "absolute",
    mode: str = "manifest",
    force: bool = False,
    voc_year: str | None = None,
    voc_masks_dirname: str = "SegmentationClass",
    cityscapes_label_type: str = "labelTrainIds",
) -> Path:
    from_format = str(from_format).strip().lower()
    path_type = str(path_type).strip().lower()
    mode = str(mode).strip().lower()
    if path_type not in ("absolute", "relative"):
        raise ValueError("--path-type must be absolute|relative")
    if mode not in ("manifest", "symlink", "copy"):
        raise ValueError("--mode must be manifest|symlink|copy")

    root_path = Path(root)
    source: dict[str, Any] = {"from": from_format, "root": str(root_path), "split": str(split)}

    if from_format in ("voc", "pascal_voc", "pascal-voc"):
        from .datasets.pascal_voc import (
            PASCAL_VOC_IGNORE_INDEX,
            PASCAL_VOC_SEG_CLASSES_21,
            resolve_pascal_voc_root,
            iter_pascal_voc_seg_samples,
        )

        paths = resolve_pascal_voc_root(root_path, year=voc_year)
        samples_out: list[dict[str, Any]] = []
        for sample in iter_pascal_voc_seg_samples(
            root_path,
            split=split,
            year=voc_year,
            masks_dirname=voc_masks_dirname,
        ):
            samples_out.append(
                {
                    "id": str(sample.sample_id),
                    "image": _maybe_relative(sample.image_path, root=paths.root, path_type=path_type),
                    "mask": _maybe_relative(sample.mask_path, root=paths.root, path_type=path_type),
                }
            )

        dataset_name = "pascal_voc" if paths.year is None else f"pascal_voc{paths.year}"
        source.update({"year": paths.year, "dataset_root_hint": str(paths.root)})
        return write_seg_dataset_descriptor(
            output,
            dataset=dataset_name,
            split=split,
            ignore_index=PASCAL_VOC_IGNORE_INDEX,
            samples=samples_out,
            classes=PASCAL_VOC_SEG_CLASSES_21,
            path_type=path_type,
            mode=mode,
            source=source,
            force=force,
        )

    if from_format in ("cityscapes",):
        from .datasets.cityscapes import (
            CITYSCAPES_IGNORE_INDEX,
            CITYSCAPES_TRAIN_CLASSES_19,
            resolve_cityscapes_paths,
            iter_cityscapes_samples,
        )

        paths = resolve_cityscapes_paths(root_path)
        samples_out = []
        for sample in iter_cityscapes_samples(root_path, split=split, label_type=cityscapes_label_type):
            samples_out.append(
                {
                    "id": str(sample.sample_id),
                    "image": _maybe_relative(sample.image_path, root=paths.images_root, path_type=path_type),
                    "mask": _maybe_relative(sample.mask_path, root=paths.labels_root or root_path, path_type=path_type),
                }
            )

        source.update({"label_type": cityscapes_label_type, "dataset_root_hint": str(root_path)})
        return write_seg_dataset_descriptor(
            output,
            dataset="cityscapes",
            split=split,
            ignore_index=CITYSCAPES_IGNORE_INDEX,
            samples=samples_out,
            classes=CITYSCAPES_TRAIN_CLASSES_19,
            path_type=path_type,
            mode=mode,
            source=source,
            force=force,
        )

    if from_format in ("ade20k", "ade"):
        from .datasets.ade20k import ADE20K_IGNORE_INDEX, resolve_ade20k_paths, iter_ade20k_samples, load_ade20k_classes

        paths = resolve_ade20k_paths(root_path)
        samples_out = []
        for sample in iter_ade20k_samples(root_path, split=split):
            samples_out.append(
                {
                    "id": str(sample.sample_id),
                    "image": _maybe_relative(sample.image_path, root=paths.root, path_type=path_type),
                    "mask": _maybe_relative(sample.mask_path, root=paths.root, path_type=path_type),
                }
            )

        classes = load_ade20k_classes(root_path)
        source.update({"dataset_root_hint": str(paths.root)})
        return write_seg_dataset_descriptor(
            output,
            dataset="ade20k",
            split=split,
            ignore_index=ADE20K_IGNORE_INDEX,
            samples=samples_out,
            classes=classes,
            path_type=path_type,
            mode=mode,
            source=source,
            force=force,
        )

    raise ValueError(f"unsupported --from: {from_format}")

