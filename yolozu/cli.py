from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yolozu import __version__

from .cli_args import parse_image_size_arg, require_non_negative_int
from .config import simple_yaml_load


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
            return data or {}
        except Exception:
            return simple_yaml_load(text)
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        return json.loads(text)
    except Exception:
        return simple_yaml_load(text)


def _build_args_from_config(cfg: dict) -> list[str]:
    args: list[str] = []
    for key, value in cfg.items():
        if value is None:
            continue
        arg = f"--{str(key).replace('_', '-') }"
        if isinstance(value, bool):
            if value:
                args.append(arg)
            continue
        if isinstance(value, (list, tuple)):
            args.append(arg)
            args.extend([str(v) for v in value])
            continue
        args.append(arg)
        args.append(str(value))
    return args


def _cmd_train(config_path: Path, extra_args: list[str] | None = None) -> int:
    try:
        from rtdetr_pose.tools.train_minimal import main as train_main
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "yolozu train is currently supported from a source checkout (e.g. `pip install -e .`) "
            "because it depends on in-repo trainer scaffolding under rtdetr_pose/tools."
        ) from exc

    argv = ["--config", str(config_path)]
    if extra_args:
        argv.extend(list(extra_args))
    return int(train_main(argv))


def _cmd_test(config_path: Path) -> int:
    try:
        from tools.run_scenarios import main as scenarios_main
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "yolozu test is currently supported from a source checkout (e.g. `pip install -e .`) "
            "because it depends on in-repo tools/ scripts."
        ) from exc

    cfg = _load_config(config_path)
    args = _build_args_from_config(cfg)
    scenarios_main(args)
    return 0


def _cmd_doctor(output: str) -> int:
    from yolozu.doctor import write_doctor_report

    return int(write_doctor_report(output=output))


def _cmd_export(args: argparse.Namespace) -> int:
    from yolozu.export import (
        DEFAULT_PREDICTIONS_PATH,
        export_dummy_predictions,
        export_labels_predictions,
        write_predictions_json,
    )

    backend = str(getattr(args, "backend", "dummy"))

    dataset = str(args.dataset)
    if not dataset:
        raise SystemExit("--dataset is required")

    try:
        if backend == "dummy":
            payload, _run = export_dummy_predictions(
                dataset_root=dataset,
                split=str(args.split) if args.split else None,
                max_images=int(args.max_images) if args.max_images is not None else None,
                score=float(args.score),
            )
        elif backend == "labels":
            payload, _run = export_labels_predictions(
                dataset_root=dataset,
                split=str(args.split) if args.split else None,
                max_images=int(args.max_images) if args.max_images is not None else None,
                score=float(args.score),
            )
        else:
            raise SystemExit(f"unsupported --backend: {backend}")
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    output = str(args.output) if args.output else DEFAULT_PREDICTIONS_PATH
    out_path = write_predictions_json(output=output, payload=payload, force=bool(args.force))
    print(str(out_path))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    if args.validate_command == "dataset":
        from yolozu.dataset import build_manifest
        from yolozu.dataset_validator import validate_dataset_records

        manifest = build_manifest(str(args.dataset), split=str(args.split) if args.split else None)
        records = manifest.get("images") or []
        if not isinstance(records, list):
            raise SystemExit("invalid dataset manifest (expected list under 'images')")
        if args.max_images is not None:
            records = records[: int(args.max_images)]

        res = validate_dataset_records(
            records,
            strict=bool(args.strict),
            mode=str(args.mode),
            check_images=not bool(args.no_check_images),
        )
        for w in res.warnings:
            print(w, file=sys.stderr)
        if res.errors:
            for e in res.errors:
                print(e, file=sys.stderr)
            return 1
        return 0

    path = Path(str(args.path))
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to parse json: {path} ({exc})") from exc

    if args.validate_command == "predictions":
        from yolozu.predictions import validate_predictions_payload

        try:
            res = validate_predictions_payload(payload, strict=bool(args.strict))
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        for w in res.warnings:
            print(w, file=sys.stderr)
        return 0

    if args.validate_command == "instance-seg":
        from yolozu.instance_segmentation_predictions import (
            normalize_instance_segmentation_predictions_payload,
            validate_instance_segmentation_predictions_entries,
        )

        try:
            entries, _meta = normalize_instance_segmentation_predictions_payload(payload)
            res = validate_instance_segmentation_predictions_entries(entries, where="predictions")
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        for w in res.warnings:
            print(w, file=sys.stderr)
        return 0

    raise SystemExit("unknown validate command")


def _cmd_eval_instance_seg(args: argparse.Namespace) -> int:
    from yolozu.instance_segmentation_report import run_instance_segmentation_eval

    out_json, out_html = run_instance_segmentation_eval(
        dataset_root=str(args.dataset),
        split=str(args.split) if args.split else None,
        predictions=str(args.predictions),
        pred_root=str(args.pred_root) if args.pred_root else None,
        classes=str(args.classes) if args.classes else None,
        output=str(args.output),
        html=str(args.html) if args.html else None,
        title=str(args.title),
        overlays_dir=str(args.overlays_dir) if args.overlays_dir else None,
        max_overlays=int(args.max_overlays),
        overlay_sort=str(args.overlay_sort),
        overlay_max_size=int(args.overlay_max_size),
        overlay_alpha=float(args.overlay_alpha),
        min_score=float(args.min_score),
        max_images=int(args.max_images) if args.max_images is not None else None,
        diag_iou=float(args.diag_iou),
        per_image_limit=int(args.per_image_limit),
        allow_rgb_masks=bool(args.allow_rgb_masks),
    )
    print(str(out_json))
    if out_html is not None:
        print(str(out_html))
    return 0


def _cmd_onnxrt_export(args: argparse.Namespace) -> int:
    from yolozu.onnxrt_export import export_predictions_onnxrt, write_predictions_json

    try:
        payload = export_predictions_onnxrt(
            dataset_root=str(args.dataset),
            split=str(args.split) if args.split else None,
            max_images=int(args.max_images) if args.max_images is not None else None,
            onnx=str(args.onnx) if args.onnx else None,
            input_name=str(args.input_name),
            boxes_output=str(args.boxes_output),
            scores_output=str(args.scores_output),
            class_output=(str(args.class_output) if args.class_output else None),
            combined_output=(str(args.combined_output) if args.combined_output else None),
            combined_format=str(args.combined_format),
            raw_output=(str(args.raw_output) if args.raw_output else None),
            raw_format=str(args.raw_format),
            raw_postprocess=str(args.raw_postprocess),
            boxes_format=str(args.boxes_format),
            boxes_scale=str(args.boxes_scale),
            min_score=float(args.min_score),
            topk=int(args.topk),
            nms_iou=float(args.nms_iou),
            agnostic_nms=bool(args.agnostic_nms),
            imgsz=int(args.imgsz),
            dry_run=bool(args.dry_run),
            strict=bool(args.strict),
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    out_path = write_predictions_json(output=str(args.output), payload=payload, force=bool(args.force))
    print(str(out_path))
    return 0


def _cmd_predict_images(args: argparse.Namespace) -> int:
    from yolozu.predict_images import predict_images

    try:
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        out_json, out_html = predict_images(
            backend=str(args.backend),
            input_dir=str(args.input_dir),
            output=str(args.output),
            score=float(args.score),
            max_images=max_images,
            force=bool(args.force),
            glob_patterns=list(args.glob) if args.glob else None,
            overlays_dir=str(args.overlays_dir) if args.overlays_dir else None,
            html=str(args.html) if args.html else None,
            title=str(args.title),
            onnx=(str(args.onnx) if args.onnx else None),
            input_name=str(args.input_name),
            boxes_output=str(args.boxes_output),
            scores_output=str(args.scores_output),
            class_output=(str(args.class_output) if args.class_output else None),
            combined_output=(str(args.combined_output) if args.combined_output else None),
            combined_format=str(args.combined_format),
            raw_output=(str(args.raw_output) if args.raw_output else None),
            raw_format=str(args.raw_format),
            raw_postprocess=str(args.raw_postprocess),
            boxes_format=str(args.boxes_format),
            boxes_scale=str(args.boxes_scale),
            min_score=float(args.min_score),
            topk=int(args.topk),
            nms_iou=float(args.nms_iou),
            agnostic_nms=bool(args.agnostic_nms),
            imgsz=int(args.imgsz),
            dry_run=bool(args.dry_run),
            strict=bool(args.strict),
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(str(out_json))
    if out_html is not None:
        print(str(out_html))
    return 0


def _cmd_eval_coco(args: argparse.Namespace) -> int:
    import time

    from yolozu.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections
    from yolozu.dataset import build_manifest
    from yolozu.predictions import load_predictions_entries, validate_predictions_entries

    dataset_root = Path(str(args.dataset)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root

    manifest = build_manifest(dataset_root, split=str(args.split) if args.split else None)
    records = list(manifest.get("images") or [])
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    gt, index = build_coco_ground_truth(records)
    image_sizes = {image["id"]: (int(image["width"]), int(image["height"])) for image in gt["images"]}

    predictions_path = Path(str(args.predictions)).expanduser()
    if not predictions_path.is_absolute():
        predictions_path = Path.cwd() / predictions_path
    predictions = load_predictions_entries(predictions_path)
    validation = validate_predictions_entries(predictions, strict=False)
    detections = predictions_to_coco_detections(
        predictions,
        coco_index=index,
        image_sizes=image_sizes,
        bbox_format=str(args.bbox_format),
    )

    if bool(args.dry_run):
        result: dict[str, object] = {
            "metrics": {"map50_95": None, "map50": None, "map75": None, "ar100": None},
            "stats": [],
            "dry_run": True,
            "counts": {"images": int(len(records)), "detections": int(len(detections))},
            "warnings": validation.warnings,
        }
    else:
        result = evaluate_coco_map(gt, detections)
        result["warnings"] = validation.warnings

    payload: dict[str, object] = {
        "report_schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(dataset_root),
        "split": manifest.get("split"),
        "split_requested": str(args.split) if args.split else None,
        "predictions": str(predictions_path),
        "bbox_format": str(args.bbox_format),
        "max_images": int(args.max_images) if args.max_images is not None else None,
        **result,
    }

    output_path = Path(str(args.output)).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(output_path))
    return 0


def _cmd_parity(args: argparse.Namespace) -> int:
    from yolozu.predictions_parity import compare_predictions

    try:
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
        image_size = parse_image_size_arg(args.image_size, flag_name="--image-size")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    report = compare_predictions(
        reference=str(args.reference),
        candidate=str(args.candidate),
        image_size=image_size,
        max_images=max_images,
        iou_thresh=float(args.iou_thresh),
        score_atol=float(args.score_atol),
        bbox_atol=float(args.bbox_atol),
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if bool(report.get("ok")) else 2


def _cmd_resources(args: argparse.Namespace) -> int:
    from yolozu import resources

    if args.resources_command == "list":
        for p in resources.list_resource_paths():
            print(p)
        return 0

    if args.resources_command == "cat":
        text = resources.read_text(str(args.path))
        print(text, end="" if text.endswith("\n") else "\n")
        return 0

    if args.resources_command == "copy":
        out = resources.copy_to(str(args.path), output=str(args.output), force=bool(args.force))
        print(str(out))
        return 0

    raise SystemExit("unknown resources command")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yolozu")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Print environment diagnostics as JSON.")
    doctor.add_argument("--output", default="reports/doctor.json", help="Output JSON path (use - for stdout).")

    export = sub.add_parser("export", help="Export predictions.json artifacts.")
    export.add_argument(
        "--backend",
        choices=("dummy", "labels"),
        default="dummy",
        help="Export backend (dummy=1 det/image; labels=use dataset labels).",
    )
    export.add_argument("--dataset", default="data/coco128", help="YOLO-format dataset root.")
    export.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    export.add_argument("--max-images", type=int, default=50, help="Optional cap for number of images.")
    export.add_argument("--score", type=float, default=0.9, help="Score to assign to exported detections (default: 0.9).")
    export.add_argument(
        "--output",
        default=None,
        help="Predictions JSON output path (default: reports/predictions.json).",
    )
    export.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")

    predict = sub.add_parser("predict-images", help="Run folder inference and write predictions JSON + overlays + HTML.")
    predict.add_argument("--backend", choices=("dummy", "onnxrt"), default="dummy", help="Inference backend.")
    predict.add_argument("--input-dir", required=True, help="Input directory containing images.")
    predict.add_argument("--glob", action="append", default=None, help="Glob pattern(s) under input dir (repeatable).")
    predict.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    predict.add_argument("--score", type=float, default=0.9, help="Dummy score when --backend=dummy.")
    predict.add_argument("--output", default="reports/predict_images.json", help="Predictions JSON output path.")
    predict.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")
    predict.add_argument("--overlays-dir", default="reports/overlays", help="Overlay images output directory.")
    predict.add_argument("--html", default="reports/predict_images.html", help="Optional HTML report path.")
    predict.add_argument("--title", default="YOLOZU predict-images report", help="HTML title.")
    predict.add_argument("--onnx", default=None, help="Path to ONNX model (required for --backend onnxrt unless --dry-run).")
    predict.add_argument("--input-name", default="images", help="ONNX input name (default: images).")
    predict.add_argument("--boxes-output", default="boxes", help="Output name for boxes tensor (default: boxes).")
    predict.add_argument("--scores-output", default="scores", help="Output name for scores tensor (default: scores).")
    predict.add_argument("--class-output", default=None, help="Optional output name for class_id tensor.")
    predict.add_argument("--combined-output", default=None, help="Optional output name for [x1,y1,x2,y2,score,class_id].")
    predict.add_argument("--combined-format", choices=("xyxy_score_class",), default="xyxy_score_class")
    predict.add_argument("--raw-output", default=None, help="Optional output name for raw head output.")
    predict.add_argument("--raw-format", choices=("yolo_84",), default="yolo_84")
    predict.add_argument("--raw-postprocess", choices=("native", "ultralytics"), default="native")
    predict.add_argument("--boxes-format", choices=("xyxy",), default="xyxy")
    predict.add_argument("--boxes-scale", choices=("abs", "norm"), default="norm")
    predict.add_argument("--min-score", type=float, default=0.001, help="Score threshold (default: 0.001).")
    predict.add_argument("--topk", type=int, default=300, help="Top-K detections per image (default: 300).")
    predict.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU for raw output decode (default: 0.7).")
    predict.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS for raw output decode.")
    predict.add_argument("--imgsz", type=int, default=640, help="Input image size (square, default: 640).")
    predict.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")
    predict.add_argument("--strict", action="store_true", help="Strict prediction schema validation before writing.")

    eval_coco = sub.add_parser("eval-coco", help="Evaluate detections with COCOeval (optional extra: yolozu[coco]).")
    eval_coco.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    eval_coco.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    eval_coco.add_argument("--predictions", required=True, help="Predictions JSON path.")
    eval_coco.add_argument(
        "--bbox-format",
        choices=("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"),
        default="cxcywh_norm",
        help="How to interpret detection bbox fields (default: cxcywh_norm).",
    )
    eval_coco.add_argument("--dry-run", action="store_true", help="Skip COCOeval; only validate/convert predictions.")
    eval_coco.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    eval_coco.add_argument("--output", default="reports/coco_eval.json", help="Output report path.")

    parity = sub.add_parser("parity", help="Compare two predictions JSON artifacts for backend parity.")
    parity.add_argument("--reference", required=True, help="Reference predictions JSON (e.g. PyTorch).")
    parity.add_argument("--candidate", required=True, help="Candidate predictions JSON (e.g. ONNXRuntime).")
    parity.add_argument("--iou-thresh", type=float, default=0.99, help="IoU threshold for a match.")
    parity.add_argument("--score-atol", type=float, default=1e-4, help="Absolute tolerance for score differences.")
    parity.add_argument("--bbox-atol", type=float, default=1e-4, help="Absolute tolerance for bbox differences.")
    parity.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    parity.add_argument("--image-size", default=None, help="Optional fixed image size (N or W,H) to skip image reads.")

    validate = sub.add_parser("validate", help="Validate artifacts (predictions JSON, instance-seg predictions).")
    validate_sub = validate.add_subparsers(dest="validate_command", required=True)
    val_pred = validate_sub.add_parser("predictions", help="Validate predictions JSON (detections+bbox schema).")
    val_pred.add_argument("path", type=str, help="Path to predictions JSON (list or wrapper).")
    val_pred.add_argument("--strict", action="store_true", help="Strict validation (types, required keys).")

    val_is = validate_sub.add_parser("instance-seg", help="Validate instance-segmentation predictions JSON (PNG masks).")
    val_is.add_argument("path", type=str, help="Path to instance-seg predictions JSON.")

    val_ds = validate_sub.add_parser("dataset", help="Validate a YOLO-format dataset layout + labels.")
    val_ds.add_argument("dataset", type=str, help="YOLO-format dataset root (contains images/ + labels/).")
    val_ds.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    val_ds.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images checked.")
    val_ds.add_argument("--strict", action="store_true", help="Strict bbox checks (range + inside-image).")
    val_ds.add_argument("--mode", choices=("fail", "warn"), default="fail", help="fail=exit nonzero on errors; warn=always exit 0.")
    val_ds.add_argument("--no-check-images", action="store_true", help="Skip image existence/size checks.")

    eis = sub.add_parser(
        "eval-instance-seg",
        help="Evaluate instance segmentation predictions (mask mAP over PNG masks).",
    )
    eis.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    eis.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    eis.add_argument("--predictions", required=True, help="Instance segmentation predictions JSON.")
    eis.add_argument("--pred-root", default=None, help="Optional root to resolve relative prediction mask paths.")
    eis.add_argument("--classes", default=None, help="Optional classes.txt/classes.json for class_id→name.")
    eis.add_argument("--output", default="reports/instance_seg_eval.json", help="Output JSON report path.")
    eis.add_argument("--html", default=None, help="Optional HTML report path.")
    eis.add_argument("--title", default="YOLOZU instance segmentation eval report", help="HTML title.")
    eis.add_argument("--overlays-dir", default=None, help="Optional directory to write overlay images for HTML.")
    eis.add_argument("--max-overlays", type=int, default=0, help="Max overlays to render (default: 0).")
    eis.add_argument(
        "--overlay-sort",
        choices=("worst", "best", "first"),
        default="worst",
        help="How to select overlay samples (default: worst).",
    )
    eis.add_argument("--overlay-max-size", type=int, default=768, help="Max size (max(H,W)) for overlay images (default: 768).")
    eis.add_argument("--overlay-alpha", type=float, default=0.5, help="Mask overlay alpha (default: 0.5).")
    eis.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold for predictions (default: 0.0).")
    eis.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to evaluate.")
    eis.add_argument("--diag-iou", type=float, default=0.5, help="IoU threshold used for per-image diagnostics/overlay selection (default: 0.5).")
    eis.add_argument("--per-image-limit", type=int, default=100, help="How many per-image rows to store in the report/meta and HTML (default: 100).")
    eis.add_argument(
        "--allow-rgb-masks",
        action="store_true",
        help="Allow 3-channel masks (uses channel 0; intended for grayscale stored as RGB).",
    )

    onnxrt = sub.add_parser("onnxrt", help="ONNXRuntime utilities (optional extra: yolozu[onnxrt]).")
    onnxrt_sub = onnxrt.add_subparsers(dest="onnxrt_command", required=True)
    onnxrt_export = onnxrt_sub.add_parser("export", help="Run ONNXRuntime inference and export predictions JSON.")
    onnxrt_export.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    onnxrt_export.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    onnxrt_export.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    onnxrt_export.add_argument("--onnx", default=None, help="Path to ONNX model (required unless --dry-run).")
    onnxrt_export.add_argument("--input-name", default="images", help="ONNX input name (default: images).")
    onnxrt_export.add_argument("--boxes-output", default="boxes", help="Output name for boxes tensor (default: boxes).")
    onnxrt_export.add_argument("--scores-output", default="scores", help="Output name for scores tensor (default: scores).")
    onnxrt_export.add_argument("--class-output", default=None, help="Optional output name for class_id tensor (default: none).")
    onnxrt_export.add_argument(
        "--combined-output",
        default=None,
        help="Optional single output name with (N,6) or (1,N,6) entries [x1,y1,x2,y2,score,class_id].",
    )
    onnxrt_export.add_argument(
        "--combined-format",
        choices=("xyxy_score_class",),
        default="xyxy_score_class",
        help="Layout for --combined-output (default: xyxy_score_class).",
    )
    onnxrt_export.add_argument(
        "--raw-output",
        default=None,
        help="Optional single output name with raw head output (e.g., 1x84x8400) to decode + NMS.",
    )
    onnxrt_export.add_argument(
        "--raw-format",
        choices=("yolo_84",),
        default="yolo_84",
        help="Layout for --raw-output (default: yolo_84).",
    )
    onnxrt_export.add_argument(
        "--raw-postprocess",
        choices=("native", "ultralytics"),
        default="native",
        help="Postprocess for --raw-output (default: native).",
    )
    onnxrt_export.add_argument(
        "--boxes-format",
        choices=("xyxy",),
        default="xyxy",
        help="Box layout produced by the model in input-image space (default: xyxy).",
    )
    onnxrt_export.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="norm",
        help="Whether boxes are in pixels (abs) or normalized [0,1] wrt input_size (default: norm).",
    )
    onnxrt_export.add_argument("--min-score", type=float, default=0.001, help="Score threshold (no NMS).")
    onnxrt_export.add_argument("--topk", type=int, default=300, help="Keep top-K detections per image (no NMS).")
    onnxrt_export.add_argument("--nms-iou", type=float, default=0.7, help="IoU threshold for NMS when decoding raw output.")
    onnxrt_export.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS when decoding raw output.",
    )
    onnxrt_export.add_argument("--imgsz", type=int, default=640, help="Input image size (square, default: 640).")
    onnxrt_export.add_argument("--output", default="reports/predictions_onnxrt.json", help="Where to write predictions JSON.")
    onnxrt_export.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")
    onnxrt_export.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")
    onnxrt_export.add_argument("--strict", action="store_true", help="Strict prediction schema validation before writing.")

    resources_p = sub.add_parser("resources", help="Access packaged configs/schemas/protocols.")
    resources_sub = resources_p.add_subparsers(dest="resources_command", required=True)
    resources_sub.add_parser("list", help="List packaged resource paths.")
    cat = resources_sub.add_parser("cat", help="Print a packaged resource to stdout.")
    cat.add_argument("path", type=str, help="Resource path under yolozu/data (e.g., schemas/predictions.schema.json).")
    copy = resources_sub.add_parser("copy", help="Copy a packaged resource to a file path.")
    copy.add_argument("path", type=str, help="Resource path under yolozu/data.")
    copy.add_argument("--output", required=True, help="Output file path.")
    copy.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    repo_root = Path(__file__).resolve().parents[1]
    dev_enabled = (repo_root / "rtdetr_pose" / "tools" / "train_minimal.py").exists() and (
        repo_root / "tools" / "run_scenarios.py"
    ).exists()
    dev_help = "(dev) Source-checkout-only commands (train/test)."
    if not dev_enabled:
        dev_help = argparse.SUPPRESS

    dev = sub.add_parser("dev", help=dev_help)
    dev_sub = dev.add_subparsers(dest="dev_command", required=True)

    dev_train = dev_sub.add_parser("train", help="Run training using a YAML/JSON config (source checkout only).")
    dev_train.add_argument("config", type=str, help="Path to train config (e.g. configs/examples/train_setting.yaml).")
    dev_train.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to rtdetr_pose/tools/train_minimal.py (e.g. --run-contract --run-id exp01).",
    )

    dev_test = dev_sub.add_parser("test", help="Run scenario tests using a YAML/JSON config (source checkout only).")
    dev_test.add_argument("config", type=str, help="Path to test config (e.g. configs/examples/test_setting.yaml).")

    # Backward-compatible hidden aliases.
    alias_help = argparse.SUPPRESS
    alias_train = sub.add_parser("train", help=alias_help)
    alias_train.add_argument("config", type=str, help="Path to train config (e.g. configs/examples/train_setting.yaml).")
    alias_train.add_argument("train_args", nargs=argparse.REMAINDER)
    alias_test = sub.add_parser("test", help=alias_help)
    alias_test.add_argument("config", type=str, help="Path to test config (e.g. configs/examples/test_setting.yaml).")

    demo = sub.add_parser("demo", help="Run small self-contained demos (CPU-friendly).")
    demo_sub = demo.add_subparsers(dest="demo_command", required=True)

    demo_is = demo_sub.add_parser("instance-seg", help="Synthetic instance-seg eval demo (numpy + Pillow).")
    demo_is.add_argument("--run-dir", default=None, help="Run directory (default: runs/yolozu_demos/instance_seg/<utc>).")
    demo_is.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    demo_is.add_argument("--num-images", type=int, default=8, help="Number of images (default: 8).")
    demo_is.add_argument("--image-size", type=int, default=96, help="Square image size (default: 96).")
    demo_is.add_argument("--max-instances", type=int, default=2, help="Max instances per image (default: 2).")

    demo_cl = demo_sub.add_parser("continual", help="Toy continual-learning demo (requires torch; CPU OK).")
    demo_cl.add_argument("--output", default=None, help="Output JSON path or dir (default: runs/yolozu_demos/continual/...).")
    demo_cl.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    demo_cl.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    demo_cl.add_argument("--method", default="ewc_replay", choices=("naive", "ewc", "replay", "ewc_replay"))
    demo_cl.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=("naive", "ewc", "replay", "ewc_replay"),
        help="Run multiple methods and write a suite report.",
    )
    demo_cl.add_argument(
        "--compare",
        action="store_true",
        help="Convenience flag: run all methods (naive/ewc/replay/ewc_replay) and write a suite report.",
    )
    demo_cl.add_argument(
        "--markdown",
        action="store_true",
        help="Also write a markdown summary table next to the JSON output (suite or single).",
    )
    demo_cl.add_argument("--steps-a", type=int, default=200, help="Training steps on domain A (default: 200).")
    demo_cl.add_argument("--steps-b", type=int, default=200, help="Training steps on domain B (default: 200).")
    demo_cl.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    demo_cl.add_argument("--hidden", type=int, default=32, help="Hidden units (default: 32).")
    demo_cl.add_argument("--lr", type=float, default=1e-2, help="Learning rate (default: 1e-2).")
    demo_cl.add_argument("--corr", type=float, default=2.0, help="Spurious correlation magnitude (default: 2.0).")
    demo_cl.add_argument("--noise", type=float, default=0.6, help="Feature noise std (default: 0.6).")
    demo_cl.add_argument("--n-train", type=int, default=4096, help="Train samples per domain (default: 4096).")
    demo_cl.add_argument("--n-eval", type=int, default=1024, help="Eval samples per domain (default: 1024).")
    demo_cl.add_argument("--ewc-lambda", type=float, default=20.0, help="EWC penalty weight (default: 20.0).")
    demo_cl.add_argument("--fisher-batches", type=int, default=64, help="Batches for Fisher estimate (default: 64).")
    demo_cl.add_argument("--replay-capacity", type=int, default=512, help="Replay buffer capacity (default: 512).")
    demo_cl.add_argument("--replay-k", type=int, default=64, help="Replay samples per step (default: 64).")

    args = parser.parse_args(argv)
    if args.command == "train":
        if not dev_enabled:
            raise SystemExit(
                "yolozu train is supported only from a source checkout (e.g. `pip install -e .`). "
                "Use `yolozu dev train <config>` when working in this repo."
            )
        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"config not found: {config_path}")
        return _cmd_train(config_path, extra_args=list(getattr(args, "train_args", []) or []))
    if args.command == "test":
        if not dev_enabled:
            raise SystemExit(
                "yolozu test is supported only from a source checkout (e.g. `pip install -e .`). "
                "Use `yolozu dev test <config>` when working in this repo."
            )
        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"config not found: {config_path}")
        return _cmd_test(config_path)
    if args.command == "dev":
        if not dev_enabled:
            raise SystemExit(
                "yolozu dev is supported only from a source checkout (e.g. `pip install -e .`). "
                "The pip-installed CLI intentionally hides dev-only commands."
            )
        if args.dev_command == "train":
            config_path = Path(args.config)
            if not config_path.exists():
                raise SystemExit(f"config not found: {config_path}")
            return _cmd_train(config_path, extra_args=list(getattr(args, "train_args", []) or []))
        if args.dev_command == "test":
            config_path = Path(args.config)
            if not config_path.exists():
                raise SystemExit(f"config not found: {config_path}")
            return _cmd_test(config_path)
        raise SystemExit("unknown dev command")
    if args.command == "doctor":
        return _cmd_doctor(str(args.output))
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "predict-images":
        return _cmd_predict_images(args)
    if args.command == "eval-coco":
        return _cmd_eval_coco(args)
    if args.command == "parity":
        return _cmd_parity(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "eval-instance-seg":
        return _cmd_eval_instance_seg(args)
    if args.command == "onnxrt":
        if args.onnxrt_command == "export":
            return _cmd_onnxrt_export(args)
        raise SystemExit("unknown onnxrt command")
    if args.command == "resources":
        return _cmd_resources(args)
    if args.command == "demo":
        if args.demo_command == "instance-seg":
            from yolozu.demos.instance_seg import run_instance_seg_demo

            out = run_instance_seg_demo(
                run_dir=args.run_dir,
                seed=int(args.seed),
                num_images=int(args.num_images),
                image_size=int(args.image_size),
                max_instances=int(args.max_instances),
            )
            try:
                payload = json.loads(Path(out).read_text(encoding="utf-8"))
                res = payload.get("result", {})
                print(f"instance-seg demo: mAP50-95={res.get('map50_95'):.3f} mAP50={res.get('map50'):.3f}")
            except Exception:
                pass
            print(str(out))
            return 0

        if args.demo_command == "continual":
            from yolozu.demos.continual import (
                format_continual_demo_suite_markdown,
                run_continual_demo,
                run_continual_demo_suite,
            )

            methods = None
            if args.methods:
                methods = [str(m) for m in args.methods]
            elif args.compare:
                methods = ["naive", "ewc", "replay", "ewc_replay"]

            if methods and len(methods) > 1:
                out = run_continual_demo_suite(
                    methods=methods,
                    output=args.output,
                    seed=int(args.seed),
                    device=str(args.device),
                    steps_a=int(args.steps_a),
                    steps_b=int(args.steps_b),
                    batch_size=int(args.batch_size),
                    hidden=int(args.hidden),
                    lr=float(args.lr),
                    corr=float(args.corr),
                    noise=float(args.noise),
                    n_train=int(args.n_train),
                    n_eval=int(args.n_eval),
                    ewc_lambda=float(args.ewc_lambda),
                    fisher_batches=int(args.fisher_batches),
                    replay_capacity=int(args.replay_capacity),
                    replay_k=int(args.replay_k),
                )
                try:
                    payload = json.loads(Path(out).read_text(encoding="utf-8"))
                    md = format_continual_demo_suite_markdown(payload)
                    print(md, end="")
                    if args.markdown:
                        md_path = Path(out).with_suffix(".md")
                        md_path.write_text(md, encoding="utf-8")
                        print(str(md_path))
                except Exception:
                    pass
                print(str(out))
                return 0

            method = str(args.method)
            if methods and len(methods) == 1:
                method = str(methods[0])

            out = run_continual_demo(
                output=args.output,
                seed=int(args.seed),
                device=str(args.device),
                method=method,
                steps_a=int(args.steps_a),
                steps_b=int(args.steps_b),
                batch_size=int(args.batch_size),
                hidden=int(args.hidden),
                lr=float(args.lr),
                corr=float(args.corr),
                noise=float(args.noise),
                n_train=int(args.n_train),
                n_eval=int(args.n_eval),
                ewc_lambda=float(args.ewc_lambda),
                fisher_batches=int(args.fisher_batches),
                replay_capacity=int(args.replay_capacity),
                replay_k=int(args.replay_k),
            )
            try:
                payload = json.loads(Path(out).read_text(encoding="utf-8"))
                metrics = payload.get("metrics", {})
                a = metrics.get("after_task_a", {})
                b = metrics.get("after_task_b", {})
                forgetting = metrics.get("forgetting_acc_a")
                gain = metrics.get("gain_acc_b")
                print(
                    f"continual demo ({method}): "
                    f"accA {a.get('acc_a'):.3f}→{b.get('acc_a'):.3f} "
                    f"accB {a.get('acc_b'):.3f}→{b.get('acc_b'):.3f} "
                    f"forget={forgetting:.3f} gain={gain:.3f}"
                )
                if args.markdown:
                    md = format_continual_demo_suite_markdown({"runs": [{"method": method, "metrics": metrics}]})
                    md_path = Path(out).with_suffix(".md")
                    md_path.write_text(md, encoding="utf-8")
                    print(str(md_path))
            except Exception:
                pass
            print(str(out))
            return 0

    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
