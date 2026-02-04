import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.boxes import cxcywh_norm_to_xyxy_abs, iou_xyxy_abs
from yolozu.image_size import get_image_size
from yolozu.predictions import load_predictions_entries


@dataclass(frozen=True)
class _Det:
    class_id: int
    score: float
    bbox: dict


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="Reference predictions JSON (e.g. PyTorch).")
    p.add_argument("--candidate", required=True, help="Candidate predictions JSON (e.g. ONNXRuntime).")
    p.add_argument(
        "--bbox-format",
        choices=("cxcywh_norm",),
        default="cxcywh_norm",
        help="BBox format stored in both JSONs (default: cxcywh_norm).",
    )
    p.add_argument("--iou-thresh", type=float, default=0.99, help="IoU threshold to consider a match.")
    p.add_argument("--score-atol", type=float, default=1e-4, help="Absolute tolerance for score differences.")
    p.add_argument("--bbox-atol", type=float, default=1e-4, help="Absolute tolerance for bbox cx/cy/w/h differences.")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    p.add_argument(
        "--image-size",
        default=None,
        help="Optional fixed image size (e.g. 640 or 640,640) to avoid reading image files.",
    )
    return p.parse_args(argv)


def _parse_image_size(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    raw = value.replace("x", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("--image-size expects 'N' or 'W,H'")


def _load_index(path: str) -> dict[str, list[_Det]]:
    entries = load_predictions_entries(path)
    out: dict[str, list[_Det]] = {}
    for e in entries:
        image = str(e.get("image", ""))
        dets = []
        for d in (e.get("detections") or []):
            if not isinstance(d, dict):
                continue
            if "class_id" not in d or "bbox" not in d or "score" not in d:
                continue
            dets.append(_Det(class_id=int(d["class_id"]), score=float(d["score"]), bbox=d["bbox"]))
        out[image] = dets
        base = image.split("/")[-1]
        if base and base not in out:
            out[base] = dets
    return out


def _bbox_tuple_norm(bbox: dict) -> tuple[float, float, float, float]:
    return float(bbox["cx"]), float(bbox["cy"]), float(bbox["w"]), float(bbox["h"])


def _close(a: float, b: float, atol: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(float(a) - float(b)) <= float(atol)


def _match_image(
    *,
    image_path: str,
    ref: list[_Det],
    cand: list[_Det],
    iou_thresh: float,
    score_atol: float,
    bbox_atol: float,
    image_size: tuple[int, int] | None,
) -> dict:
    if image_size is None:
        w, h = get_image_size(image_path)
    else:
        w, h = image_size
    ref_xyxy = [cxcywh_norm_to_xyxy_abs(_bbox_tuple_norm(d.bbox), width=w, height=h) for d in ref]
    cand_xyxy = [cxcywh_norm_to_xyxy_abs(_bbox_tuple_norm(d.bbox), width=w, height=h) for d in cand]

    used = set()
    matches = []
    failures = []

    for i, r in enumerate(ref):
        best = None
        best_iou = -1.0
        for j, c in enumerate(cand):
            if j in used:
                continue
            if c.class_id != r.class_id:
                continue
            iou = iou_xyxy_abs(ref_xyxy[i], cand_xyxy[j])
            if iou > best_iou:
                best_iou = iou
                best = j

        if best is None or best_iou < float(iou_thresh):
            failures.append(
                {
                    "type": "missing_match",
                    "ref_index": i,
                    "class_id": r.class_id,
                    "ref_score": r.score,
                    "best_iou": None if best is None else best_iou,
                }
            )
            continue

        used.add(best)
        c = cand[best]

        rb = _bbox_tuple_norm(r.bbox)
        cb = _bbox_tuple_norm(c.bbox)
        bbox_ok = all(_close(a, b, bbox_atol) for a, b in zip(rb, cb))
        score_ok = _close(r.score, c.score, score_atol)
        matches.append(
            {
                "ref_index": i,
                "cand_index": best,
                "class_id": r.class_id,
                "iou": best_iou,
                "score_ref": r.score,
                "score_cand": c.score,
                "score_ok": score_ok,
                "bbox_ok": bbox_ok,
            }
        )
        if not (bbox_ok and score_ok):
            failures.append(
                {
                    "type": "value_mismatch",
                    "ref_index": i,
                    "cand_index": best,
                    "class_id": r.class_id,
                    "iou": best_iou,
                    "ref": {"score": r.score, "bbox": r.bbox},
                    "cand": {"score": c.score, "bbox": c.bbox},
                }
            )

    extras = [j for j in range(len(cand)) if j not in used]
    return {
        "image": image_path,
        "size": {"width": w, "height": h},
        "counts": {"ref": len(ref), "cand": len(cand), "matched": len(matches), "extra_cand": len(extras)},
        "matches": matches,
        "extras": extras,
        "failures": failures,
        "ok": len(failures) == 0,
    }


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    image_size = _parse_image_size(args.image_size)

    ref_idx = _load_index(args.reference)
    cand_idx = _load_index(args.candidate)

    images = sorted({k for k in ref_idx.keys() if "/" in k or Path(k).exists()})
    if args.max_images is not None:
        images = images[: max(0, int(args.max_images))]
    if not images:
        raise SystemExit("no comparable images found in reference predictions (need absolute paths or existing files)")

    per_image = []
    ok = True
    for image in images:
        ref = ref_idx.get(image, [])
        cand = cand_idx.get(image, [])
        result = _match_image(
            image_path=image,
            ref=ref,
            cand=cand,
            iou_thresh=args.iou_thresh,
            score_atol=args.score_atol,
            bbox_atol=args.bbox_atol,
            image_size=image_size,
        )
        per_image.append(result)
        ok = ok and bool(result["ok"])

    report = {
        "reference": args.reference,
        "candidate": args.candidate,
        "bbox_format": args.bbox_format,
        "iou_thresh": args.iou_thresh,
        "score_atol": args.score_atol,
        "bbox_atol": args.bbox_atol,
        "images": len(per_image),
        "ok": ok,
        "results": per_image,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

