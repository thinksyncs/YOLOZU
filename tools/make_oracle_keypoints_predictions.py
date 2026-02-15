import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit oracle keypoints predictions from dataset GT labels.")
    p.add_argument("--dataset", required=True, help="YOLOZU dataset root (with keypoints in labels).")
    p.add_argument("--split", default=None, help="Split (default: auto).")
    p.add_argument("--output", default="reports/pred_keypoints_oracle.json", help="Output predictions JSON path.")
    p.add_argument("--score", type=float, default=1.0, help="Score to assign to oracle detections (default: 1.0).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images.")
    return p.parse_args(argv)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = Path(str(args.dataset))
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest.get("images") or []
    if not isinstance(records, list):
        records = []
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    predictions = []
    for rec in records:
        image = rec.get("image")
        if not isinstance(image, str) or not image:
            continue
        dets = []
        for lab in rec.get("labels") or []:
            if not isinstance(lab, dict):
                continue
            if lab.get("keypoints") is None:
                continue
            dets.append(
                {
                    "class_id": int(lab.get("class_id", 0)),
                    "score": float(args.score),
                    "bbox": {"cx": float(lab["cx"]), "cy": float(lab["cy"]), "w": float(lab["w"]), "h": float(lab["h"])},
                    "keypoints": lab.get("keypoints"),
                }
            )
        predictions.append({"image": image, "detections": dets})

    out_path = Path(str(args.output))
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "predictions": predictions,
        "meta": {
            "created_at": _now_utc(),
            "source": "oracle_from_gt",
            "dataset": str(dataset_root),
            "split": manifest.get("split"),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

