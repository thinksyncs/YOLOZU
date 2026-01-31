import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.image_size import get_image_size
from yolozu.letterbox import compute_letterbox, input_xyxy_to_orig_xyxy, orig_xyxy_to_cxcywh_norm
from yolozu.predictions import validate_predictions_entries


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="YOLO-format COCO root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--onnx", default=None, help="Path to ONNX model (required unless --dry-run).")
    p.add_argument("--input-name", default="images", help="ONNX input name (default: images).")
    p.add_argument("--boxes-output", default="boxes", help="Output name for boxes tensor (default: boxes).")
    p.add_argument("--scores-output", default="scores", help="Output name for scores tensor (default: scores).")
    p.add_argument("--class-output", default=None, help="Optional output name for class_id tensor (default: none).")
    p.add_argument(
        "--boxes-format",
        choices=("xyxy",),
        default="xyxy",
        help="Box layout produced by the model in input-image space (default: xyxy).",
    )
    p.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="norm",
        help="Whether boxes are in pixels (abs) or normalized [0,1] (norm) wrt input_size (default: norm).",
    )
    p.add_argument("--min-score", type=float, default=0.001, help="Score threshold (no NMS).")
    p.add_argument("--topk", type=int, default=300, help="Keep top-K detections per image (no NMS).")
    p.add_argument("--output", default="reports/predictions_onnxrt.json", help="Where to write predictions JSON.")
    p.add_argument("--wrap", action="store_true", help="Wrap as {predictions:[...], meta:{...}}.")
    p.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")
    p.add_argument("--strict", action="store_true", help="Strict prediction schema validation before writing.")
    return p.parse_args(argv)


def _now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_boxes_and_scores(outputs: dict[str, object], *, boxes_key: str, scores_key: str, class_key: str | None):
    boxes = outputs.get(boxes_key)
    scores = outputs.get(scores_key)
    if boxes is None:
        raise ValueError(f"missing boxes output: {boxes_key}")
    if scores is None:
        raise ValueError(f"missing scores output: {scores_key}")
    class_ids = outputs.get(class_key) if class_key else None
    return boxes, scores, class_ids


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    input_size = 640  # pinned (YOLO26 protocol)
    dataset_root = repo_root / args.dataset
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    predictions = []
    if args.dry_run:
        for record in records:
            predictions.append({"image": record["image"], "detections": []})
        validate_predictions_entries(predictions, strict=args.strict)
        payload = (
            {
                "predictions": predictions,
                "meta": {
                    "timestamp": _now_utc(),
                    "exporter": "onnxruntime",
                    "dry_run": True,
                    "protocol_id": "yolo26",
                    "imgsz": input_size,
                    "dataset": args.dataset,
                    "split": manifest["split"],
                    "max_images": args.max_images,
                    "note": "dry-run mode: no inference performed",
                },
            }
            if args.wrap
            else predictions
        )
        out_path = repo_root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(out_path)
        return

    if not args.onnx:
        raise SystemExit("--onnx is required unless --dry-run is set")

    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for onnxruntime exporter") from exc

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required (pip install onnxruntime)") from exc

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for image loading (pip install pillow)") from exc

    model_path = Path(args.onnx)
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    if not model_path.exists():
        raise SystemExit(f"onnx model not found: {model_path}")

    providers = None
    try:
        providers = ort.get_available_providers()
    except Exception:
        providers = None

    sess = ort.InferenceSession(str(model_path), providers=providers)

    def preprocess(image_path: str):
        w, h = get_image_size(image_path)
        letterbox = compute_letterbox(orig_w=w, orig_h=h, input_size=input_size)

        img = Image.open(image_path).convert("RGB")
        img = img.resize((letterbox.new_w, letterbox.new_h), resample=Image.BILINEAR)

        canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
        canvas.paste(img, (int(round(letterbox.pad_x)), int(round(letterbox.pad_y))))

        x = np.asarray(canvas, dtype=np.float32) / 255.0  # (H,W,C)
        x = np.transpose(x, (2, 0, 1))  # (C,H,W)
        x = np.expand_dims(x, axis=0)  # (1,C,H,W)
        return x, (w, h), letterbox

    for record in records:
        image_path = record["image"]
        x, (orig_w, orig_h), letterbox = preprocess(image_path)

        raw_outputs = sess.run(None, {args.input_name: x})
        names = [o.name for o in sess.get_outputs()]
        outputs = dict(zip(names, raw_outputs))
        boxes_t, scores_t, class_t = _resolve_boxes_and_scores(
            outputs, boxes_key=args.boxes_output, scores_key=args.scores_output, class_key=args.class_output
        )

        boxes = np.asarray(boxes_t)
        scores = np.asarray(scores_t)
        class_ids = None if class_t is None else np.asarray(class_t)

        if scores.ndim == 2:
            # scores: (N, C) -> pick best class
            class_ids = np.argmax(scores, axis=1)
            scores = np.max(scores, axis=1)
        elif scores.ndim != 1:
            raise ValueError(f"unsupported scores shape: {scores.shape}")

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"unsupported boxes shape: {boxes.shape}")
        if class_ids is None:
            raise ValueError("class ids missing: provide --class-output or use (N,C) scores")

        scores = scores.astype(float)
        class_ids = class_ids.astype(int)

        idx = [i for i, s in enumerate(scores.tolist()) if float(s) >= float(args.min_score)]
        idx.sort(key=lambda i: float(scores[i]), reverse=True)
        idx = idx[: max(0, int(args.topk))]

        detections = []
        for i in idx:
            b = boxes[i].tolist()
            if args.boxes_format != "xyxy":
                raise ValueError("only --boxes-format xyxy is supported in this skeleton")

            if args.boxes_scale == "norm":
                x1, y1, x2, y2 = (float(b[0]) * input_size, float(b[1]) * input_size, float(b[2]) * input_size, float(b[3]) * input_size)
            else:
                x1, y1, x2, y2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

            orig_xyxy = input_xyxy_to_orig_xyxy((x1, y1, x2, y2), letterbox=letterbox, orig_w=orig_w, orig_h=orig_h)
            bbox = orig_xyxy_to_cxcywh_norm(orig_xyxy, orig_w=orig_w, orig_h=orig_h)

            detections.append({"class_id": int(class_ids[i]), "score": float(scores[i]), "bbox": bbox})

        predictions.append({"image": image_path, "detections": detections})

    validate_predictions_entries(predictions, strict=args.strict)

    meta = {
        "timestamp": _now_utc(),
        "exporter": "onnxruntime",
        "protocol_id": "yolo26",
        "imgsz": input_size,
        "dataset": args.dataset,
        "split": manifest["split"],
        "max_images": args.max_images,
        "onnx": str(model_path),
        "onnx_sha256": _sha256(model_path),
        "input_name": args.input_name,
        "boxes_output": args.boxes_output,
        "scores_output": args.scores_output,
        "class_output": args.class_output,
        "boxes_format": args.boxes_format,
        "boxes_scale": args.boxes_scale,
        "min_score": args.min_score,
        "topk": args.topk,
        "providers": providers,
        "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }

    payload = {"predictions": predictions, "meta": meta} if args.wrap else predictions
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(out_path)


if __name__ == "__main__":
    main()

