import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections
from yolozu.dataset import build_manifest
from yolozu.eval_protocol import apply_eval_protocol_args, load_eval_protocol
from yolozu.predictions import load_predictions_payload, validate_predictions_entries


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=("yolo26",),
        default=None,
        help="Apply canonical evaluation protocol presets (pins split/bbox_format).",
    )
    parser.add_argument("--dataset", default="data/coco128", help="YOLO-format COCO root (images/ + labels/).")
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto (val2017 if present else train2017).",
    )
    parser.add_argument(
        "--predictions-glob",
        required=True,
        help="Glob for predictions JSON files (e.g. 'reports/pred_yolo26*.json').",
    )
    parser.add_argument(
        "--bbox-format",
        choices=("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"),
        default="cxcywh_norm",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip COCOeval and only validate/convert predictions (no pycocotools required).",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--strict", action="store_true", help="Strict prediction schema validation.")
    parser.add_argument("--output", default="reports/eval_suite.json", help="Output JSON path.")
    return parser.parse_args(argv)


def _resolve_args(argv):
    args = _parse_args(argv)
    protocol = load_eval_protocol(args.protocol) if args.protocol else None
    if protocol:
        args = apply_eval_protocol_args(args, protocol)
    return args, protocol


def _now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_image_size(value: Any) -> dict[str, int] | None:
    if isinstance(value, dict):
        w = _coerce_int(value.get("width"))
        h = _coerce_int(value.get("height"))
        if w is None or h is None:
            return None
        if w <= 0 or h <= 0:
            return None
        return {"width": int(w), "height": int(h)}
    if isinstance(value, (list, tuple)) and len(value) == 2:
        w = _coerce_int(value[0])
        h = _coerce_int(value[1])
        if w is None or h is None:
            return None
        if w <= 0 or h <= 0:
            return None
        return {"width": int(w), "height": int(h)}
    return None


def _resolve_meta_config(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    run = meta.get("run")
    if isinstance(run, dict):
        cfg = run.get("config_fingerprint")
        if isinstance(cfg, dict):
            return cfg
    return meta


def _extract_preprocess(entries: list[dict[str, Any]], meta_cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    if entries:
        entry0 = entries[0]
        pp = entry0.get("preprocess")
        if isinstance(pp, dict):
            out: dict[str, Any] = {}
            method = pp.get("method")
            if isinstance(method, str) and method:
                out["method"] = method
            norm = pp.get("normalize")
            if isinstance(norm, str) and norm:
                out["normalize"] = norm
            input_size = _normalize_image_size(pp.get("input_size"))
            if input_size is None:
                input_size = _normalize_image_size(entry0.get("image_size"))
            if input_size is not None:
                out["input_size"] = input_size
            if out:
                return out

    exporter = None
    if isinstance(meta_cfg, dict):
        v = meta_cfg.get("exporter")
        if isinstance(v, str) and v:
            exporter = v
    if exporter in {"onnxruntime", "tensorrt"}:
        return {
            "method": "letterbox",
            "input_color": "RGB",
            "normalize": "0_1",
            "resize_interp": "linear",
            "letterbox_fill": [114, 114, 114],
        }
    return None


def _extract_export_settings(
    entries: list[dict[str, Any]], meta: dict[str, Any] | None
) -> dict[str, Any] | None:
    meta_cfg = _resolve_meta_config(meta)

    imgsz = None
    if isinstance(meta_cfg, dict):
        imgsz = _coerce_int(meta_cfg.get("imgsz"))
    if imgsz is None and entries:
        entry0 = entries[0]
        size = _normalize_image_size(entry0.get("image_size"))
        if size is None and isinstance(entry0.get("preprocess"), dict):
            size = _normalize_image_size(entry0["preprocess"].get("input_size"))
        if size is not None and size["width"] == size["height"]:
            imgsz = int(size["width"])

    score_threshold = None
    max_detections = None
    if isinstance(meta_cfg, dict):
        score_threshold = _coerce_float(
            meta_cfg.get("score_threshold", meta_cfg.get("conf", meta_cfg.get("min_score")))
        )
        max_detections = _coerce_int(
            meta_cfg.get("max_detections", meta_cfg.get("max_det", meta_cfg.get("topk")))
        )

    image_size = None
    if imgsz is not None:
        image_size = {"width": int(imgsz), "height": int(imgsz)}
    if entries:
        entry0 = entries[0]
        inferred = _normalize_image_size(entry0.get("image_size"))
        if inferred is None and isinstance(entry0.get("preprocess"), dict):
            inferred = _normalize_image_size(entry0["preprocess"].get("input_size"))
        if inferred is not None:
            image_size = inferred

    preprocess = _extract_preprocess(entries, meta_cfg)

    settings: dict[str, Any] = {
        "imgsz": imgsz,
        "image_size": image_size,
        "score_threshold": score_threshold,
        "max_detections": max_detections,
        "preprocess": preprocess,
    }
    if all(v is None for v in settings.values()):
        return None
    return settings


def _extract_predictions_meta_ref(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    meta_cfg = _resolve_meta_config(meta) or {}
    ref: dict[str, Any] = {}
    run = meta.get("run")
    if isinstance(run, dict):
        cfg_hash = run.get("config_hash")
        if isinstance(cfg_hash, str) and cfg_hash:
            ref["config_hash"] = cfg_hash

    for key in (
        "exporter",
        "generator",
        "protocol_id",
        "adapter",
        "config",
        "checkpoint",
        "onnx",
        "onnx_sha256",
        "engine",
        "engine_sha256",
    ):
        value = meta_cfg.get(key)
        if value is None:
            value = meta.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            ref[key] = value
    return ref or None


def main(argv=None):
    args, protocol = _resolve_args(sys.argv[1:] if argv is None else argv)

    dataset_root = repo_root / args.dataset
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    split_effective = manifest["split"]
    if args.max_images is not None:
        records = records[: args.max_images]

    gt, index = build_coco_ground_truth(records)
    image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt["images"]}

    if Path(args.predictions_glob).is_absolute():
        pred_paths = [Path(p) for p in sorted(glob.glob(args.predictions_glob))]
    else:
        pred_paths = sorted((repo_root / ".").glob(args.predictions_glob))
    if not pred_paths:
        raise SystemExit(f"no predictions matched: {args.predictions_glob}")

    results = []
    for path in pred_paths:
        entries, meta = load_predictions_payload(path)
        validation = validate_predictions_entries(entries, strict=args.strict)
        dt = predictions_to_coco_detections(
            entries, coco_index=index, image_sizes=image_sizes, bbox_format=args.bbox_format
        )
        export_settings = _extract_export_settings(entries, meta)
        meta_ref = _extract_predictions_meta_ref(meta)
        if args.dry_run:
            eval_result = {
                "metrics": {
                    "map50_95": None,
                    "map50": None,
                    "map75": None,
                    "ar100": None,
                },
                "stats": [],
                "dry_run": True,
                "counts": {"images": len(records), "detections": len(dt)},
            }
        else:
            eval_result = evaluate_coco_map(gt, dt)
        results.append(
            {
                "name": path.stem,
                "path": str(path),
                "warnings": validation.warnings,
                "export_settings": export_settings,
                "predictions_meta_ref": meta_ref,
                **eval_result,
            }
        )

    payload = {
        "report_schema_version": 1,
        "timestamp": _now_utc(),
        "protocol_id": args.protocol,
        "protocol": protocol,
        "dataset": str(args.dataset),
        "split": split_effective,
        "split_requested": args.split,
        "bbox_format": args.bbox_format,
        "images": len(records),
        "results": results,
    }

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
