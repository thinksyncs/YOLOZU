import argparse
import json
import os
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert COCO person keypoints annotations to YOLOZU YOLO-format labels.")
    p.add_argument("--coco-root", required=True, help="COCO root containing val2017/ and annotations/.")
    p.add_argument(
        "--annotations",
        default="annotations/person_keypoints_val2017.json",
        help="Path to person_keypoints_*.json (relative to --coco-root).",
    )
    p.add_argument("--images-dir", default="val2017", help="Images directory under --coco-root (default: val2017).")
    p.add_argument("--out", required=True, help="Output dataset root (writes labels/ + dataset.json).")
    p.add_argument("--out-split", default="val2017", help="Output split name under labels/ (default: val2017).")
    p.add_argument("--min-kps", type=int, default=1, help="Require at least N labeled keypoints (default: 1).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    p.add_argument("--link-images", action="store_true", help="Symlink images into out/images/<split>/ (optional).")
    return p.parse_args(argv)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _bbox_xywh_to_cxcywh_norm(bbox_xywh: list[float], *, width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx = (x + w / 2.0) / float(width)
    cy = (y + h / 2.0) / float(height)
    bw = w / float(width)
    bh = h / float(height)
    return float(cx), float(cy), float(bw), float(bh)


def _kps_to_norm_triplets(kps: list[float], *, width: int, height: int) -> list[float]:
    # COCO keypoints: [x1,y1,v1, x2,y2,v2, ...] in pixels.
    out: list[float] = []
    if len(kps) % 3 != 0:
        return out
    for i in range(0, len(kps), 3):
        x = float(kps[i + 0]) / float(width)
        y = float(kps[i + 1]) / float(height)
        v = float(kps[i + 2])
        out.extend([x, y, v])
    return out


def _count_labeled(kps: list[float]) -> int:
    if len(kps) % 3 != 0:
        return 0
    cnt = 0
    for i in range(0, len(kps), 3):
        v = kps[i + 2]
        try:
            if int(v) > 0:
                cnt += 1
        except Exception:
            continue
    return cnt


def main(argv: list[str] | None = None) -> int:
    import sys

    args = _parse_args(sys.argv[1:] if argv is None else argv)

    coco_root = Path(str(args.coco_root))
    if not coco_root.is_absolute():
        coco_root = Path.cwd() / coco_root
    ann_path = coco_root / str(args.annotations)
    images_dir = coco_root / str(args.images_dir)
    if not ann_path.exists():
        raise SystemExit(f"annotations not found: {ann_path}")
    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")

    out_root = Path(str(args.out))
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root

    out_split = str(args.out_split)
    out_labels = out_root / "labels" / out_split
    out_images = out_root / "images" / out_split
    _ensure_dir(out_labels)
    if bool(args.link_images):
        _ensure_dir(out_images)

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images") or []
    annotations = payload.get("annotations") or []
    categories = payload.get("categories") or []

    person_cat_id = None
    for cat in categories:
        if isinstance(cat, dict) and cat.get("name") == "person":
            try:
                person_cat_id = int(cat.get("id"))
            except Exception:
                person_cat_id = None
            break
    if person_cat_id is None:
        raise SystemExit("person category not found in annotations")

    images_by_id: dict[int, dict] = {}
    for im in images:
        if not isinstance(im, dict):
            continue
        try:
            images_by_id[int(im["id"])] = im
        except Exception:
            continue

    ann_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        try:
            if int(ann.get("category_id")) != int(person_cat_id):
                continue
        except Exception:
            continue
        try:
            image_id = int(ann.get("image_id"))
        except Exception:
            continue
        ann_by_image.setdefault(image_id, []).append(ann)

    written = 0
    for image_id, im in sorted(images_by_id.items(), key=lambda kv: kv[0]):
        if args.max_images is not None and written >= int(args.max_images):
            break
        file_name = im.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue
        try:
            width = int(im.get("width"))
            height = int(im.get("height"))
        except Exception:
            continue
        src_img = images_dir / file_name
        if not src_img.exists():
            continue

        anns = ann_by_image.get(int(image_id), [])
        lines: list[str] = []
        for ann in anns:
            bbox = ann.get("bbox")
            kps = ann.get("keypoints")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            if not (isinstance(kps, list) and len(kps) >= 3):
                continue
            if _count_labeled([float(v) for v in kps]) < int(args.min_kps):
                continue
            cx, cy, bw, bh = _bbox_xywh_to_cxcywh_norm([float(v) for v in bbox], width=width, height=height)
            # YOLOZU keypoints: append x y v triplets (normalized).
            kps_norm = _kps_to_norm_triplets([float(v) for v in kps], width=width, height=height)
            if not kps_norm:
                continue
            # class_id=0 for person (keypoints task).
            parts = [f"{0:d}", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"] + [f"{v:.6f}" for v in kps_norm]
            lines.append(" ".join(parts))

        stem = Path(file_name).stem
        (out_labels / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if bool(args.link_images):
            dst = out_images / src_img.name
            if not dst.exists():
                os.symlink(src_img, dst)

        written += 1

    # Prefer dataset.json so we don't need to copy images.
    dataset_json = {
        "images_dir": str(images_dir),
        "labels_dir": str(out_labels),
        "split": out_split,
        "task": "keypoints",
        "source": str(ann_path),
        "notes": "person-only; class_id=0; keypoints appended as x y v triplets (normalized).",
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

