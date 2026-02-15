import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a COCO128-derived dataset with pose/depth sidecars for testing.")
    p.add_argument("--in-dataset", default="data/coco128", help="Input dataset root (YOLO images/labels).")
    p.add_argument("--out-dataset", required=True, help="Output dataset root.")
    p.add_argument("--split", default="train2017", help="Split under images/ and labels/ (default: train2017).")
    p.add_argument(
        "--link-images",
        action="store_true",
        help="Symlink images instead of copying (recommended on RunPod).",
    )
    p.add_argument(
        "--pose-mode",
        choices=("identity", "bbox_yaw"),
        default="bbox_yaw",
        help="How to synthesize GT rotation (default: bbox_yaw).",
    )
    p.add_argument(
        "--z-mode",
        choices=("area", "constant"),
        default="area",
        help="How to synthesize GT depth z (default: area).",
    )
    p.add_argument("--z-base", type=float, default=1.0, help="Base z for z-mode (default: 1.0).")
    p.add_argument("--z-scale", type=float, default=2.0, help="Scale for z-mode=area (default: 2.0).")
    p.add_argument("--fx", type=float, default=640.0, help="Synthetic intrinsics fx (default: 640).")
    p.add_argument("--fy", type=float, default=640.0, help="Synthetic intrinsics fy (default: 640).")
    p.add_argument("--cx", type=float, default=320.0, help="Synthetic intrinsics cx (default: 320).")
    p.add_argument("--cy", type=float, default=320.0, help="Synthetic intrinsics cy (default: 320).")
    p.add_argument("--seed", type=int, default=0, help="Seed (reserved; deterministic synthesis).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to include.")
    return p.parse_args(argv)


def _rot_z(theta: float) -> list[list[float]]:
    c = math.cos(theta)
    s = math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _write_file(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def _copy_or_link(src: Path, dst: Path, *, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    in_root = Path(str(args.in_dataset))
    if not in_root.is_absolute():
        in_root = repo_root / in_root
    out_root = Path(str(args.out_dataset))
    if not out_root.is_absolute():
        out_root = repo_root / out_root

    split = str(args.split)
    manifest = build_manifest(in_root, split=split)
    records = manifest.get("images") or []
    if not isinstance(records, list):
        raise SystemExit(f"invalid manifest: {in_root}")

    if args.max_images is not None:
        records = records[: int(args.max_images)]

    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for record in records:
        image = Path(str(record.get("image", "")))
        if not image.is_absolute():
            image = in_root / image
        if not image.exists():
            continue
        name = image.name
        stem = image.stem

        # Copy/symlink image.
        _copy_or_link(image, out_images / name, link=bool(args.link_images))

        # Copy label txt.
        src_label = in_root / "labels" / split / f"{stem}.txt"
        dst_label = out_labels / f"{stem}.txt"
        if src_label.exists() and not dst_label.exists():
            shutil.copy2(src_label, dst_label)

        labels = record.get("labels") or []
        if not isinstance(labels, list):
            labels = []

        t_list: list[list[float]] = []
        r_list: list[list[list[float]]] = []
        off_list: list[list[float]] = []
        for lab in labels:
            if not isinstance(lab, dict):
                continue
            try:
                # YOLOZU label dict uses top-level cx/cy/w/h, but keep bbox dict
                # fallback for other producers.
                bb = lab.get("bbox") if isinstance(lab.get("bbox"), dict) else {}
                cx = float(lab.get("cx", bb.get("cx")))
                cy = float(lab.get("cy", bb.get("cy")))
                w = float(lab.get("w", bb.get("w")))
                h = float(lab.get("h", bb.get("h")))
            except Exception:
                continue

            if str(args.z_mode) == "constant":
                z = float(args.z_base)
            else:
                z = float(args.z_base) + float(args.z_scale) * float(max(0.0, w) * max(0.0, h))

            # Simple translation: x/y from normalized center (arbitrary units), z as above.
            x = (cx - 0.5) * z
            y = (cy - 0.5) * z
            t_list.append([float(x), float(y), float(z)])

            if str(args.pose_mode) == "identity":
                r_list.append([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            else:
                yaw = (cx - 0.5) * math.pi
                r_list.append(_rot_z(float(yaw)))

            off_list.append([0.0, 0.0])

        meta = {
            "K_gt": [float(args.fx), float(args.fy), float(args.cx), float(args.cy)],
            "t_gt": t_list,
            "R_gt": r_list,
            "offsets_gt": off_list,
        }
        _write_file(out_labels / f"{stem}.json", json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
