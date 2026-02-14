import argparse
import json
import os
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.datasets.pascal_voc import (  # noqa: E402
    PASCAL_VOC_IGNORE_INDEX,
    PASCAL_VOC_SEG_CLASSES_21,
    iter_pascal_voc_seg_samples,
    resolve_pascal_voc_root,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Pascal VOC semantic segmentation layout + dataset.json.")
    p.add_argument(
        "--voc-root",
        required=True,
        help="VOCdevkit root or VOC year root (e.g. /path/to/VOCdevkit or /path/to/VOCdevkit/VOC2012).",
    )
    p.add_argument("--year", default=None, help="VOC year selector when --voc-root points to VOCdevkit (e.g. 2012).")
    p.add_argument("--split", choices=("train", "val", "trainval", "test"), default="train", help="Split to prepare.")
    p.add_argument(
        "--masks-dirname",
        default="SegmentationClass",
        help="Mask directory name under the VOC year root (default: SegmentationClass).",
    )
    p.add_argument("--out", required=True, help="Output root (will write dataset.json).")
    p.add_argument(
        "--mode",
        choices=("manifest", "symlink", "copy"),
        default="manifest",
        help="manifest: only dataset.json with absolute paths; symlink/copy: create images/ + masks/ layout under --out.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files under --out (symlink/copy only).")
    return p.parse_args(argv)


def _ensure_empty(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"refusing to overwrite existing path (use --force): {path}")
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)


def _copy_or_link(src: Path, dst: Path, *, mode: str, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            raise SystemExit(f"refusing to overwrite existing file (use --force): {dst}")
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    raise ValueError(f"unknown mode: {mode}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    resolved = resolve_pascal_voc_root(args.voc_root, year=args.year)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_json_path = out_root / "dataset.json"

    mode = str(args.mode)
    split = str(args.split)
    if mode != "manifest":
        _ensure_empty(out_root / "images" / split, force=bool(args.force))
        _ensure_empty(out_root / "masks" / split, force=bool(args.force))

    samples = list(
        iter_pascal_voc_seg_samples(
            resolved.root,
            split=split,
            year=resolved.year,
            masks_dirname=str(args.masks_dirname),
        )
    )

    out_samples: list[dict] = []
    for s in samples:
        img_out = str(s.image_path)
        mask_out = str(s.mask_path) if s.mask_path is not None else None
        if mode != "manifest":
            img_rel = Path("images") / split / s.image_path.name
            img_dst = out_root / img_rel
            _copy_or_link(s.image_path, img_dst, mode=mode, force=bool(args.force))
            img_out = str(img_rel)

            if s.mask_path is not None:
                mask_rel = Path("masks") / split / s.mask_path.name
                mask_dst = out_root / mask_rel
                _copy_or_link(s.mask_path, mask_dst, mode=mode, force=bool(args.force))
                mask_out = str(mask_rel)

        out_samples.append(
            {
                "id": s.sample_id,
                "image": img_out,
                "mask": mask_out,
            }
        )

    payload = {
        "dataset": "pascal_voc",
        "task": "semantic_segmentation",
        "year": resolved.year,
        "split": split,
        "masks_dirname": str(args.masks_dirname),
        "mode": mode,
        "path_type": ("absolute" if mode == "manifest" else "relative"),
        "ignore_index": int(PASCAL_VOC_IGNORE_INDEX),
        "classes": list(PASCAL_VOC_SEG_CLASSES_21),
        "samples": out_samples,
    }
    dataset_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(out_root)
    print(dataset_json_path)


if __name__ == "__main__":
    main()

