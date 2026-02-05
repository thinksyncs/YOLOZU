import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import DummyAdapter, RTDETRPoseAdapter
from yolozu.dataset import build_manifest
from yolozu.predictions_transform import apply_tta
from yolozu.tta.config import TTTConfig
from yolozu.tta.integration import run_ttt


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        choices=("dummy", "rtdetr_pose"),
        default="dummy",
        help="Which adapter to run (default: dummy).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="YOLO-format dataset root (defaults to data/coco128).",
    )
    parser.add_argument(
        "--config",
        default="rtdetr_pose/configs/base.json",
        help="Config path for rtdetr_pose adapter.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for rtdetr_pose adapter (default: cpu).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        help="Image size for rtdetr_pose adapter (one value or two values).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Score threshold for rtdetr_pose adapter (default: 0.3).",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=50,
        help="Max detections per image for rtdetr_pose adapter (default: 50).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint for rtdetr_pose adapter.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for number of images (for quick smoke runs).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto.",
    )
    parser.add_argument(
        "--output",
        default="reports/predictions.json",
        help="Where to write predictions JSON.",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap output as {predictions: [...], meta: {...}} (recommended).",
    )
    parser.add_argument("--tta", action="store_true", help="Enable TTA post-transform on predictions.")
    parser.add_argument("--tta-seed", type=int, default=None, help="Seed for TTA randomness.")
    parser.add_argument("--tta-flip-prob", type=float, default=0.5, help="Flip probability for TTA.")
    parser.add_argument("--tta-norm-only", action="store_true", help="Update only normalized bbox values for TTA.")
    parser.add_argument("--tta-log-out", default=None, help="Optional path to write TTA log JSON.")

    parser.add_argument("--ttt", action="store_true", help="Enable test-time training (TTT) before inference.")
    parser.add_argument(
        "--ttt-preset",
        choices=("safe", "adapter_only", "mim_safe"),
        default=None,
        help="Recommended TTT presets that override method/steps/lr/filter. Choices: safe, adapter_only, mim_safe.",
    )
    parser.add_argument(
        "--ttt-method",
        choices=("tent", "mim"),
        default="tent",
        help="TTT method (default: tent).",
    )
    parser.add_argument("--ttt-steps", type=int, default=1, help="Total TTT steps to run (default: 1).")
    parser.add_argument("--ttt-batch-size", type=int, default=1, help="TTT batch size (default: 1).")
    parser.add_argument("--ttt-lr", type=float, default=1e-4, help="TTT learning rate (default: 1e-4).")
    parser.add_argument(
        "--ttt-update-filter",
        choices=("all", "norm_only", "adapter_only"),
        default="all",
        help="Which parameters to update during TTT (default: all).",
    )
    parser.add_argument(
        "--ttt-include",
        action="append",
        default=None,
        help="Only update parameters whose name contains this substring (repeatable).",
    )
    parser.add_argument(
        "--ttt-exclude",
        action="append",
        default=None,
        help="Exclude parameters whose name contains this substring (repeatable).",
    )
    parser.add_argument(
        "--ttt-max-batches",
        type=int,
        default=1,
        help="Cap number of distinct batches used for TTT (default: 1).",
    )
    parser.add_argument("--ttt-seed", type=int, default=None, help="Optional RNG seed for TTT.")
    parser.add_argument("--ttt-mask-prob", type=float, default=0.6, help="MIM mask probability (default: 0.6).")
    parser.add_argument("--ttt-patch-size", type=int, default=16, help="MIM patch size (default: 16).")
    parser.add_argument("--ttt-mask-value", type=float, default=0.0, help="MIM mask fill value (default: 0.0).")
    parser.add_argument("--ttt-log-out", default=None, help="Optional path to write TTT log JSON.")
    return parser.parse_args(argv)


def _apply_ttt_preset(args):
    preset = getattr(args, "ttt_preset", None)
    if not preset:
        return
    if preset == "safe":
        args.ttt_method = "tent"
        args.ttt_steps = 1
        args.ttt_batch_size = 1
        args.ttt_lr = 1e-4
        args.ttt_update_filter = "norm_only"
        args.ttt_max_batches = 1
    elif preset == "adapter_only":
        args.ttt_method = "tent"
        args.ttt_steps = 1
        args.ttt_batch_size = 1
        args.ttt_lr = 1e-4
        args.ttt_update_filter = "adapter_only"
        args.ttt_max_batches = 1
    elif preset == "mim_safe":
        args.ttt_method = "mim"
        args.ttt_steps = 1
        args.ttt_batch_size = 1
        args.ttt_lr = 1e-4
        args.ttt_update_filter = "adapter_only"
        args.ttt_max_batches = 1
    else:
        raise SystemExit(f"unknown preset: {preset}")


def _summarize_tta(predictions, *, warnings):
    total = 0
    applied = 0
    for entry in predictions:
        mask = entry.get("tta_mask") if isinstance(entry, dict) else None
        if isinstance(mask, list):
            total += len(mask)
            applied += sum(1 for flag in mask if flag)
    ratio = float(applied) / float(total) if total else 0.0
    return {
        "detections": int(total),
        "applied": int(applied),
        "applied_ratio": float(ratio),
        "warnings": list(warnings),
    }


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    _apply_ttt_preset(args)

    dataset_root = Path(args.dataset) if args.dataset else (repo_root / "data" / "coco128")
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    if args.adapter == "dummy":
        adapter = DummyAdapter()
    else:
        image_size = None
        if args.image_size:
            if len(args.image_size) == 1:
                image_size = (args.image_size[0], args.image_size[0])
            elif len(args.image_size) == 2:
                image_size = (args.image_size[0], args.image_size[1])
            else:
                raise SystemExit("--image-size expects 1 or 2 integers")
        adapter = RTDETRPoseAdapter(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            image_size=image_size or (320, 320),
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
        )

    ttt_report = None
    if args.ttt:
        ttt_config = TTTConfig(
            enabled=True,
            method=str(args.ttt_method),
            steps=int(args.ttt_steps),
            batch_size=int(args.ttt_batch_size),
            lr=float(args.ttt_lr),
            update_filter=str(args.ttt_update_filter),
            include=list(args.ttt_include) if args.ttt_include else None,
            exclude=list(args.ttt_exclude) if args.ttt_exclude else None,
            max_batches=int(args.ttt_max_batches),
            seed=args.ttt_seed,
            log_out=args.ttt_log_out,
            mim_mask_prob=float(args.ttt_mask_prob),
            mim_patch_size=int(args.ttt_patch_size),
            mim_mask_value=float(args.ttt_mask_value),
        )
        try:
            ttt_report = run_ttt(adapter, records, config=ttt_config).to_dict()
        except Exception as exc:
            extra = ""
            try:
                from yolozu.tta.ttt_mim import select_parameters

                if hasattr(adapter, "get_model"):
                    model = adapter.get_model()
                else:
                    model = None
                if model is not None:
                    params = select_parameters(
                        model,
                        update_filter=str(ttt_config.update_filter),
                        include=ttt_config.include,
                        exclude=ttt_config.exclude,
                    )
                    count = 0
                    seen = set()
                    for p in params:
                        pid = id(p)
                        if pid in seen:
                            continue
                        seen.add(pid)
                        count += int(p.numel())
                    extra = (
                        f" (method={ttt_config.method} update_filter={ttt_config.update_filter} "
                        f"selected_param_count={count} steps={ttt_config.steps} lr={ttt_config.lr})"
                    )
            except Exception:
                extra = ""
            raise SystemExit(f"TTT failed: {exc}{extra}")

    predictions = adapter.predict(records)

    tta_warnings = []
    tta_summary = None
    if args.tta:
        tta = apply_tta(
            predictions,
            enabled=True,
            seed=args.tta_seed,
            flip_prob=args.tta_flip_prob,
            norm_only=bool(args.tta_norm_only),
        )
        predictions = tta.entries
        tta_warnings = tta.warnings
        tta_summary = _summarize_tta(predictions, warnings=tta_warnings)

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.wrap:
        payload = {
            "predictions": predictions,
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "adapter": args.adapter,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "images": len(records),
                "tta": {
                    "enabled": bool(args.tta),
                    "seed": args.tta_seed,
                    "flip_prob": float(args.tta_flip_prob),
                    "norm_only": bool(args.tta_norm_only),
                    "warnings": tta_warnings,
                    "summary": tta_summary,
                },
                "ttt": {
                    "enabled": bool(args.ttt),
                    "preset": args.ttt_preset,
                    "method": str(args.ttt_method),
                    "steps": int(args.ttt_steps),
                    "batch_size": int(args.ttt_batch_size),
                    "lr": float(args.ttt_lr),
                    "update_filter": str(args.ttt_update_filter),
                    "include": list(args.ttt_include) if args.ttt_include else None,
                    "exclude": list(args.ttt_exclude) if args.ttt_exclude else None,
                    "max_batches": int(args.ttt_max_batches),
                    "seed": args.ttt_seed,
                    "mim": {
                        "mask_prob": float(args.ttt_mask_prob),
                        "patch_size": int(args.ttt_patch_size),
                        "mask_value": float(args.ttt_mask_value),
                    },
                    "report": ttt_report,
                },
            },
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    else:
        output_path.write_text(json.dumps(predictions, indent=2, sort_keys=True))

    print(output_path)

    if args.tta_log_out and args.tta:
        log_path = Path(args.tta_log_out)
        if not log_path.is_absolute():
            log_path = repo_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output": str(output_path),
            "tta": {
                "enabled": bool(args.tta),
                "seed": args.tta_seed,
                "flip_prob": float(args.tta_flip_prob),
                "norm_only": bool(args.tta_norm_only),
                "summary": tta_summary,
            },
        }
        log_path.write_text(json.dumps(log_payload, indent=2, sort_keys=True))
        print(log_path)

    if args.ttt_log_out and args.ttt:
        log_path = Path(args.ttt_log_out)
        if not log_path.is_absolute():
            log_path = repo_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output": str(output_path),
            "ttt": {
                "enabled": bool(args.ttt),
                "method": str(args.ttt_method),
                "steps": int(args.ttt_steps),
                "batch_size": int(args.ttt_batch_size),
                "lr": float(args.ttt_lr),
                "update_filter": str(args.ttt_update_filter),
                "include": list(args.ttt_include) if args.ttt_include else None,
                "exclude": list(args.ttt_exclude) if args.ttt_exclude else None,
                "max_batches": int(args.ttt_max_batches),
                "seed": args.ttt_seed,
                "mim": {
                    "mask_prob": float(args.ttt_mask_prob),
                    "patch_size": int(args.ttt_patch_size),
                    "mask_value": float(args.ttt_mask_value),
                },
                "report": ttt_report,
            },
        }
        log_path.write_text(json.dumps(log_payload, indent=2, sort_keys=True))
        print(log_path)


if __name__ == "__main__":
    main()
