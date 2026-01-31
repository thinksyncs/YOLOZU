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
from yolozu.predictions import validate_predictions_entries


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="YOLO-format COCO root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--engine", default=None, help="Path to TensorRT engine plan (required unless --dry-run).")
    p.add_argument("--output", default="reports/predictions_trt.json", help="Where to write predictions JSON.")
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


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    input_size = 640  # pinned (YOLO26 protocol)
    dataset_root = repo_root / args.dataset
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    predictions = [{"image": record["image"], "detections": []} for record in records]
    validate_predictions_entries(predictions, strict=args.strict)

    if not args.dry_run:
        if not args.engine:
            raise SystemExit("--engine is required unless --dry-run is set")
        # This is intentionally a skeleton to keep the repo Apache-2.0-friendly and dependency-light.
        # A real implementation should:
        # - load TensorRT engine (tensorrt)
        # - allocate bindings (pycuda or cuda-python)
        # - apply deterministic preprocessing (imgsz=640, letterbox)
        # - decode model outputs into YOLOZU prediction schema (no NMS)
        raise RuntimeError(
            "TensorRT exporter skeleton: install tensorrt + CUDA bindings in your env, then implement engine execution/decoding."
        )

    engine_path = None
    if args.engine:
        engine_path = Path(args.engine)
        if not engine_path.is_absolute():
            engine_path = repo_root / engine_path

    meta = {
        "timestamp": _now_utc(),
        "exporter": "tensorrt",
        "dry_run": bool(args.dry_run),
        "protocol_id": "yolo26",
        "imgsz": input_size,
        "dataset": args.dataset,
        "split": manifest["split"],
        "max_images": args.max_images,
        "engine": None if engine_path is None else str(engine_path),
        "engine_sha256": None if engine_path is None or not engine_path.exists() else _sha256(engine_path),
        "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "note": "dry-run mode: no inference performed",
    }

    payload = {"predictions": predictions, "meta": meta} if args.wrap else predictions
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(out_path)


if __name__ == "__main__":
    main()

