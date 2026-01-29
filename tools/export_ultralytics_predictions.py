import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.boxes import xyxy_abs_to_cxcywh_norm
from yolozu.dataset import build_manifest
from yolozu.image_size import get_image_size


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ultralytics model path/name (e.g. yolo26s.pt).")
    parser.add_argument("--dataset", default="data/coco128", help="YOLO-format dataset root.")
    parser.add_argument("--split", default=None, help="Dataset split (e.g. val2017). Default: auto.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640).")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001).")
    parser.add_argument("--device", default=None, help="Device string passed to Ultralytics (e.g. cpu, 0).")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--output", default="reports/predictions_ultralytics.json", help="Output predictions JSON.")
    parser.add_argument("--wrap", action="store_true", help="Wrap output as {predictions: [...], meta: {...}}.")
    return parser.parse_args(argv)


def _load_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ultralytics is required. Install it in your inference environment (e.g. `python3 -m pip install ultralytics`)."
        ) from exc
    return YOLO


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    YOLO = _load_ultralytics()
    model = YOLO(args.model)

    dataset_root = repo_root / args.dataset
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    predictions = []
    for record in records:
        image_path = Path(record["image"])
        width, height = get_image_size(image_path)

        predict_kwargs = {
            "source": str(image_path),
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "verbose": False,
        }
        if args.device is not None:
            predict_kwargs["device"] = args.device

        results = model.predict(**predict_kwargs)
        if not results:
            predictions.append({"image": str(image_path), "detections": []})
            continue

        res0 = results[0]
        dets = []
        boxes = getattr(res0, "boxes", None)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy = boxes.xyxy
            conf = getattr(boxes, "conf", None)
            cls = getattr(boxes, "cls", None)

            n = int(xyxy.shape[0]) if hasattr(xyxy, "shape") else len(xyxy)
            for i in range(n):
                x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                cx, cy, w, h = xyxy_abs_to_cxcywh_norm((x1, y1, x2, y2), width=width, height=height)
                dets.append(
                    {
                        "class_id": int(cls[i].item()) if cls is not None else 0,
                        "score": float(conf[i].item()) if conf is not None else 0.0,
                        "bbox": {"cx": cx, "cy": cy, "w": w, "h": h},
                    }
                )

        predictions.append({"image": str(image_path), "detections": dets})

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.wrap:
        payload = {
            "predictions": predictions,
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": args.model,
                "dataset": args.dataset,
                "split": args.split,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "device": args.device,
                "images": len(records),
            },
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    else:
        output_path.write_text(json.dumps(predictions, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()

