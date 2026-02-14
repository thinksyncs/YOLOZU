import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or name (e.g., yolo26n.pt)")
    parser.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/)")
    parser.add_argument("--split", default=None, help="Dataset split (default: auto-detect)")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source override (directory or file). Defaults to dataset images/<split>.",
    )
    parser.add_argument("--output", required=True, help="Where to write predictions JSON")
    parser.add_argument("--image-size", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001)")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold (default: 0.7)")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image (default: 300)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for inference (default: 1)")
    parser.add_argument("--device", default="cuda", help="Device for inference (default: cuda)")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference where supported")
    parser.add_argument(
        "--end2end",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use end2end (NMS-free) head when supported (default: true)",
    )
    return parser.parse_args(argv)


def _result_path(result):
    for key in ("path", "orig_path"):
        value = getattr(result, key, None)
        if value:
            return str(value)
    return None


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise SystemExit("ultralytics package is required (pip install ultralytics)") from exc

    manifest = build_manifest(args.dataset, split=args.split)
    records = manifest["images"]
    image_paths = [record["image"] for record in records]
    images_dir = Path(args.dataset) / "images" / manifest["split"]
    source = args.source or str(images_dir)
    if not Path(source).exists():
        raise SystemExit(f"source not found: {source}")

    model = YOLO(args.model)
    results = model.predict(
        source=source,
        imgsz=int(args.image_size),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        batch=int(args.batch),
        device=args.device,
        half=bool(args.half),
        end2end=bool(args.end2end),
        stream=True,
        verbose=False,
    )

    outputs = []
    for result in results:
        image_path = _result_path(result)
        if image_path is None:
            # Fallback to sequential mapping if result doesn't expose a path.
            if image_paths:
                image_path = image_paths[len(outputs)]
            else:
                image_path = ""

        dets = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn
            conf = boxes.conf
            cls = boxes.cls
            if xywhn is not None and conf is not None and cls is not None:
                xywhn_list = xywhn.detach().cpu().tolist()
                conf_list = conf.detach().cpu().tolist()
                cls_list = cls.detach().cpu().tolist()
                for bbox, score, class_id in zip(xywhn_list, conf_list, cls_list):
                    if len(bbox) != 4:
                        continue
                    dets.append(
                        {
                            "class_id": int(class_id),
                            "score": float(score),
                            "bbox": {
                                "cx": float(bbox[0]),
                                "cy": float(bbox[1]),
                                "w": float(bbox[2]),
                                "h": float(bbox[3]),
                            },
                        }
                    )

        outputs.append({"image": image_path, "detections": dets})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
