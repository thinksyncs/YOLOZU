import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a BOP dataset split into a YOLOZU YOLO-format dataset with sidecars.")
    p.add_argument("--bop-root", required=True, help="Path to extracted BOP dataset root (e.g., /tmp/bop/tless).")
    p.add_argument("--split", required=True, help="BOP split folder name (e.g., train_primesense, val, test).")
    p.add_argument("--out", required=True, help="Output dataset root (YOLO images/ + labels/).")
    p.add_argument("--out-split", default="train2017", help="Output split name under images/ and labels/ (default: train2017).")
    p.add_argument("--bbox-source", choices=("bbox_vis", "bbox_obj"), default="bbox_vis", help="Which BOP bbox field to use.")
    p.add_argument("--visib-fract-min", type=float, default=0.0, help="Minimum visibility fraction (default: 0.0).")
    p.add_argument("--max-scenes", type=int, default=None, help="Optional cap for scenes to convert.")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for images to convert.")
    p.add_argument("--link-images", action="store_true", help="Symlink images instead of copying (recommended).")
    p.add_argument(
        "--class-map",
        default="obj_id_minus_1",
        choices=("obj_id_minus_1",),
        help="Class id mapping (default: obj_id_minus_1).",
    )
    p.add_argument("--t-scale", type=float, default=0.001, help="Scale for BOP translation units to meters (default: 0.001 for mm->m).")
    return p.parse_args(argv)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_or_link(src: Path, dst: Path, *, link: bool) -> None:
    _ensure_dir(dst.parent)
    if dst.exists():
        return
    if link:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def _find_image(scene_dir: Path, subdir: str, image_id: int) -> Path | None:
    stem = f"{image_id:06d}"
    for ext in (".png", ".jpg", ".jpeg"):
        cand = scene_dir / subdir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return int(w), int(h)


def _bbox_xywh_to_cxcywh_norm(bbox_xywh: list[float], *, width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx = (x + w / 2.0) / float(width)
    cy = (y + h / 2.0) / float(height)
    bw = w / float(width)
    bh = h / float(height)
    return float(cx), float(cy), float(bw), float(bh)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    bop_root = Path(str(args.bop_root))
    if not bop_root.is_absolute():
        bop_root = Path.cwd() / bop_root
    split_dir = bop_root / str(args.split)
    if not split_dir.exists():
        raise SystemExit(f"split not found: {split_dir}")

    out_root = Path(str(args.out))
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root

    out_split = str(args.out_split)
    out_images = out_root / "images" / out_split
    out_labels = out_root / "labels" / out_split
    _ensure_dir(out_images)
    _ensure_dir(out_labels)

    scene_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    if args.max_scenes is not None:
        scene_dirs = scene_dirs[: int(args.max_scenes)]

    converted_images = 0
    for scene_dir in scene_dirs:
        gt_path = scene_dir / "scene_gt.json"
        cam_path = scene_dir / "scene_camera.json"
        info_path = scene_dir / "scene_gt_info.json"
        if not (gt_path.exists() and cam_path.exists() and info_path.exists()):
            continue

        scene_gt = _load_json(gt_path)
        scene_cam = _load_json(cam_path)
        scene_info = _load_json(info_path)

        image_ids = sorted([int(k) for k in scene_gt.keys()])
        for image_id in image_ids:
            if args.max_images is not None and converted_images >= int(args.max_images):
                break

            rgb_path = _find_image(scene_dir, "rgb", image_id)
            if rgb_path is None:
                continue
            depth_path = _find_image(scene_dir, "depth", image_id)

            width, height = _image_size(rgb_path)

            instances = scene_gt.get(str(image_id)) or []
            infos = scene_info.get(str(image_id)) or []
            cam = scene_cam.get(str(image_id)) or {}
            k = cam.get("cam_K")
            if not isinstance(k, list) or len(k) < 9:
                continue

            # Copy/link RGB.
            out_name = f"{scene_dir.name}_{image_id:06d}{rgb_path.suffix.lower()}"
            out_img = out_images / out_name
            _copy_or_link(rgb_path, out_img, link=bool(args.link_images))

            label_lines: list[str] = []
            t_list: list[list[float]] = []
            r_list: list[list[list[float]]] = []
            off_list: list[list[float]] = []

            for inst, info in zip(instances, infos):
                if not isinstance(inst, dict) or not isinstance(info, dict):
                    continue
                try:
                    obj_id = int(inst.get("obj_id"))
                except Exception:
                    continue

                visib = info.get("visib_fract")
                if visib is not None and float(visib) < float(args.visib_fract_min):
                    continue

                bbox = info.get(str(args.bbox_source))
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                if float(bbox[2]) <= 1.0 or float(bbox[3]) <= 1.0:
                    continue

                cx, cy, bw, bh = _bbox_xywh_to_cxcywh_norm(bbox, width=width, height=height)
                if bw <= 0.0 or bh <= 0.0:
                    continue

                if args.class_map == "obj_id_minus_1":
                    class_id = int(obj_id) - 1
                else:
                    raise SystemExit(f"unsupported class_map: {args.class_map}")

                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                r = inst.get("cam_R_m2c")
                t = inst.get("cam_t_m2c")
                if not (isinstance(r, list) and len(r) == 9 and isinstance(t, list) and len(t) == 3):
                    # Skip pose lists if missing, but keep label.
                    continue

                r3 = [
                    [float(r[0]), float(r[1]), float(r[2])],
                    [float(r[3]), float(r[4]), float(r[5])],
                    [float(r[6]), float(r[7]), float(r[8])],
                ]
                t3 = [float(t[0]) * float(args.t_scale), float(t[1]) * float(args.t_scale), float(t[2]) * float(args.t_scale)]
                r_list.append(r3)
                t_list.append(t3)
                off_list.append([0.0, 0.0])

            if not label_lines:
                converted_images += 1
                continue

            stem = Path(out_name).stem
            (out_labels / f"{stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            sidecar: dict[str, Any] = {
                "K_gt": [float(k[i]) for i in range(9)],
                "depth_path": (str(depth_path) if depth_path is not None else None),
                "depth_scale": cam.get("depth_scale"),
                "scene_id": int(scene_dir.name),
                "image_id": int(image_id),
                "bop_root": str(bop_root),
                "bop_split": str(args.split),
            }
            if r_list and t_list:
                sidecar["R_gt"] = r_list
                sidecar["t_gt"] = t_list
                sidecar["offsets_gt"] = off_list

            (out_labels / f"{stem}.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            converted_images += 1

        if args.max_images is not None and converted_images >= int(args.max_images):
            break

    (out_root / "dataset.json").write_text(
        json.dumps(
            {"images_dir": str(out_images), "labels_dir": str(out_labels), "split": out_split},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

