import argparse
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.benchmark import measure_latency
from yolozu.coco_keypoints_eval import build_coco_keypoints_ground_truth, evaluate_coco_oks_map, predictions_to_coco_keypoints
from yolozu.dataset import build_manifest
from yolozu.keypoints_eval import evaluate_keypoints_pck
from yolozu.metrics_report import build_report, write_json
from yolozu.predictions import load_predictions_entries, load_predictions_index
from yolozu.run_record import build_run_record


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark keypoints evaluation (PCK + optional OKS mAP).")
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--predictions", required=True, help="Predictions JSON (detections may include keypoints).")
    p.add_argument("--output", default="reports/benchmark_keypoints_eval.json", help="Output JSON report path.")

    p.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching det→GT (default: 0.5).")
    p.add_argument("--pck-threshold", type=float, default=0.1, help="PCK threshold (default: 0.1).")
    p.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold for predictions (default: 0.0).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to evaluate.")

    p.add_argument("--oks", action="store_true", help="Also include COCO OKS mAP (requires pycocotools).")
    p.add_argument("--oks-sigmas", default=None, help="OKS sigmas: 'coco17' or comma-separated floats (len=K).")
    p.add_argument("--oks-sigmas-file", default=None, help="JSON file containing list[float] sigmas (len=K).")
    p.add_argument("--oks-max-dets", type=int, default=20, help="COCOeval maxDets for keypoints (default: 20).")

    p.add_argument("--warmup", type=int, default=1, help="Benchmark warmup iterations (default: 1).")
    p.add_argument("--iterations", type=int, default=5, help="Benchmark iterations (default: 5).")
    return p.parse_args(argv)


def _resolve(value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return repo_root / p


def _parse_oks_sigmas(value: str) -> list[float]:
    raw = value.strip().lower()
    if raw in ("coco", "coco17", "coco-17"):
        from yolozu.coco_keypoints_eval import COCO17_KPT_OKS_SIGMAS

        return list(COCO17_KPT_OKS_SIGMAS)
    parts = [p.strip() for p in value.replace(" ", ",").split(",") if p.strip()]
    return [float(p) for p in parts]


def _resolve_oks_sigmas(args: argparse.Namespace, *, keypoints_count: int) -> list[float]:
    if args.oks_sigmas_file:
        path = _resolve(str(args.oks_sigmas_file))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--oks-sigmas-file must contain a JSON list")
        sigmas = [float(v) for v in payload]
    elif args.oks_sigmas:
        sigmas = _parse_oks_sigmas(str(args.oks_sigmas))
    else:
        from yolozu.coco_keypoints_eval import COCO17_KPT_OKS_SIGMAS

        sigmas = list(COCO17_KPT_OKS_SIGMAS) if int(keypoints_count) == 17 else []

    if len(sigmas) != int(keypoints_count):
        raise ValueError(f"OKS sigmas length mismatch: expected {keypoints_count}, got {len(sigmas)}")
    return sigmas


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = _resolve(args.dataset)
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    split_effective = manifest["split"]
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    pred_path = _resolve(args.predictions)
    pred_index = load_predictions_index(pred_path)

    def step_pck() -> None:
        evaluate_keypoints_pck(
            records=records,
            predictions_index=pred_index,
            iou_threshold=float(args.iou_threshold),
            pck_threshold=float(args.pck_threshold),
            min_score=float(args.min_score),
            per_image_limit=0,
        )

    oks_gt: dict[str, Any] | None = None
    oks_dt: list[dict[str, Any]] | None = None
    oks_sigmas: list[float] | None = None
    oks_keypoints_count: int | None = None
    if bool(args.oks):
        gt, coco_index = build_coco_keypoints_ground_truth(records, keypoints_format="xy_norm")
        if int(coco_index.keypoints_count) <= 0:
            raise SystemExit("OKS requested but dataset has no GT keypoints")
        sigmas = _resolve_oks_sigmas(args, keypoints_count=int(coco_index.keypoints_count))
        image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt.get("images", []) or []}
        preds_entries = load_predictions_entries(pred_path)
        dt = predictions_to_coco_keypoints(
            preds_entries,
            coco_index=coco_index,
            image_sizes=image_sizes,
            keypoints_format="xy_norm",
            min_score=float(args.min_score),
        )
        oks_gt = gt
        oks_dt = dt
        oks_sigmas = sigmas
        oks_keypoints_count = int(coco_index.keypoints_count)

    def step() -> None:
        step_pck()
        if oks_gt is not None and oks_dt is not None and oks_sigmas is not None:
            evaluate_coco_oks_map(
                oks_gt,
                oks_dt,
                sigmas=oks_sigmas,
                max_dets=int(args.oks_max_dets),
            )

    eval_result = evaluate_keypoints_pck(
        records=records,
        predictions_index=pred_index,
        iou_threshold=float(args.iou_threshold),
        pck_threshold=float(args.pck_threshold),
        min_score=float(args.min_score),
        per_image_limit=0,
    )
    if oks_gt is not None and oks_dt is not None and oks_sigmas is not None:
        oks = evaluate_coco_oks_map(
            oks_gt,
            oks_dt,
            sigmas=oks_sigmas,
            max_dets=int(args.oks_max_dets),
        )
        oks_metrics = oks.get("metrics") if isinstance(oks, dict) else None
        if isinstance(oks_metrics, dict):
            eval_result.setdefault("metrics", {}).update(oks_metrics)

    bench = measure_latency(iterations=int(args.iterations), warmup=int(args.warmup), step=step)

    metrics = dict(eval_result.get("metrics") if isinstance(eval_result, dict) else {})
    metrics["benchmark"] = bench

    report = build_report(
        metrics=metrics,
        meta={
            "dataset": str(args.dataset),
            "split": str(split_effective),
            "predictions": str(args.predictions),
            "run": build_run_record(repo_root=repo_root, argv=sys.argv[1:], args=vars(args), dataset_root=str(dataset_root)),
            "oks": {
                "enabled": bool(args.oks),
                "keypoints_count": oks_keypoints_count,
                "sigmas": oks_sigmas,
                "max_dets": int(args.oks_max_dets),
                "counts": {
                    "dt": None if oks_dt is None else int(len(oks_dt)),
                    "gt_annotations": None if oks_gt is None else int(len((oks_gt.get("annotations") or []))),
                },
            },
        },
    )
    out_path = _resolve(args.output)
    write_json(out_path, report)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
