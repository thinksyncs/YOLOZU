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
from yolozu.tta.presets import apply_ttt_preset_args


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
        "--lora-r",
        type=int,
        default=0,
        help="Enable LoRA by setting rank r>0 (default: 0 disables).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha scaling (default: r).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout on inputs (default: 0.0).",
    )
    parser.add_argument(
        "--lora-target",
        default="head",
        choices=("head", "all_linear", "all_conv1x1", "all_linear_conv1x1"),
        help="Where to apply LoRA (default: head).",
    )
    parser.add_argument(
        "--lora-freeze-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze base weights and train LoRA params only (default: false).",
    )
    parser.add_argument(
        "--lora-train-bias",
        choices=("none", "all"),
        default="none",
        help="If LoRA is enabled, optionally train biases too (default: none).",
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
        choices=("safe", "adapter_only", "mim_safe", "cotta_safe", "eata_safe", "sar_safe"),
        default=None,
        help="Recommended TTT presets that override core knobs (method/steps/lr/filter/max_batches) and fill safety guards unless explicitly set.",
    )
    parser.add_argument(
        "--ttt-method",
        choices=("tent", "mim", "cotta", "eata", "sar"),
        default="tent",
        help="TTT method (default: tent).",
    )
    parser.add_argument(
        "--ttt-reset",
        choices=("stream", "sample"),
        default="stream",
        help="TTT reset policy: stream keeps adapted weights; sample resets per image (default: stream).",
    )
    parser.add_argument("--ttt-steps", type=int, default=1, help="Total TTT steps to run (default: 1).")
    parser.add_argument("--ttt-batch-size", type=int, default=1, help="TTT batch size (default: 1).")
    parser.add_argument("--ttt-lr", type=float, default=1e-4, help="TTT learning rate (default: 1e-4).")
    parser.add_argument(
        "--ttt-stop-on-non-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop TTT if loss/grad/update norms become non-finite (default: true).",
    )
    parser.add_argument(
        "--ttt-rollback-on-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rollback last TTT step when a guard triggers (default: true).",
    )
    parser.add_argument("--ttt-max-grad-norm", type=float, default=None, help="Optional grad clipping norm (default: none).")
    parser.add_argument(
        "--ttt-max-update-norm",
        type=float,
        default=None,
        help="Stop if per-step weight update L2 norm exceeds this (default: none).",
    )
    parser.add_argument(
        "--ttt-max-total-update-norm",
        type=float,
        default=None,
        help="Stop if total drift from initial weights exceeds this (default: none).",
    )
    parser.add_argument(
        "--ttt-max-loss-ratio",
        type=float,
        default=None,
        help="Stop if loss exceeds (initial_loss * ratio) (default: none).",
    )
    parser.add_argument(
        "--ttt-max-loss-increase",
        type=float,
        default=None,
        help="Stop if loss exceeds (initial_loss + delta) (default: none).",
    )
    parser.add_argument(
        "--ttt-update-filter",
        choices=("all", "norm_only", "adapter_only", "lora_only", "lora_norm_only"),
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
    parser.add_argument("--ttt-cotta-ema-momentum", type=float, default=0.999, help="CoTTA EMA momentum (default: 0.999).")
    parser.add_argument(
        "--ttt-cotta-augmentations",
        action="append",
        default=None,
        help="CoTTA augmentation branch name (repeatable, e.g. identity/hflip).",
    )
    parser.add_argument(
        "--ttt-cotta-aggregation",
        choices=("confidence_weighted_mean", "mean"),
        default="confidence_weighted_mean",
        help="CoTTA logits aggregation mode (default: confidence_weighted_mean).",
    )
    parser.add_argument("--ttt-cotta-restore-prob", type=float, default=0.01, help="CoTTA stochastic restore probability (default: 0.01).")
    parser.add_argument("--ttt-cotta-restore-interval", type=int, default=1, help="CoTTA restore cadence in steps (default: 1).")
    parser.add_argument("--ttt-eata-conf-min", type=float, default=0.2, help="EATA minimum confidence threshold (default: 0.2).")
    parser.add_argument("--ttt-eata-entropy-min", type=float, default=0.05, help="EATA minimum entropy threshold (default: 0.05).")
    parser.add_argument("--ttt-eata-entropy-max", type=float, default=3.0, help="EATA maximum entropy threshold (default: 3.0).")
    parser.add_argument("--ttt-eata-min-valid-dets", type=int, default=1, help="EATA minimum valid detections per sample (default: 1).")
    parser.add_argument("--ttt-eata-anchor-lambda", type=float, default=1e-3, help="EATA anchor regularization weight (default: 1e-3).")
    parser.add_argument("--ttt-eata-selected-ratio-min", type=float, default=0.0, help="EATA minimum selected-sample ratio per step (default: 0.0).")
    parser.add_argument("--ttt-eata-max-skip-streak", type=int, default=3, help="EATA max consecutive skipped steps before stop (default: 3).")
    parser.add_argument("--ttt-sar-rho", type=float, default=0.05, help="SAR perturbation radius rho (default: 0.05).")
    parser.add_argument(
        "--ttt-sar-adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use adaptive SAR perturbation scaling by parameter magnitude (default: false).",
    )
    parser.add_argument("--ttt-sar-first-step-scale", type=float, default=1.0, help="SAR first-step scaling factor (default: 1.0).")
    parser.add_argument("--ttt-log-out", default=None, help="Optional path to write TTT log JSON.")
    return parser.parse_args(argv)


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

    apply_ttt_preset_args(args)

    if args.adapter == "dummy" and int(args.lora_r) > 0:
        raise SystemExit("--lora-* flags are only supported with --adapter rtdetr_pose")

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
            lora_r=int(args.lora_r),
            lora_alpha=(float(args.lora_alpha) if args.lora_alpha is not None else None),
            lora_dropout=float(args.lora_dropout),
            lora_target=str(args.lora_target),
            lora_freeze_base=bool(args.lora_freeze_base),
            lora_train_bias=str(args.lora_train_bias),
        )

    def _ttt_or_die(_records):
        try:
            return run_ttt(adapter, _records, config=ttt_config).to_dict()
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

    ttt_report = None
    if args.ttt:
        ttt_config = TTTConfig(
            enabled=True,
            method=str(args.ttt_method),
            reset=str(args.ttt_reset),
            steps=int(args.ttt_steps),
            batch_size=int(args.ttt_batch_size),
            lr=float(args.ttt_lr),
            stop_on_non_finite=bool(args.ttt_stop_on_non_finite),
            rollback_on_stop=bool(args.ttt_rollback_on_stop),
            max_grad_norm=(float(args.ttt_max_grad_norm) if args.ttt_max_grad_norm is not None else None),
            max_update_norm=(float(args.ttt_max_update_norm) if args.ttt_max_update_norm is not None else None),
            max_total_update_norm=(
                float(args.ttt_max_total_update_norm) if args.ttt_max_total_update_norm is not None else None
            ),
            max_loss_ratio=(float(args.ttt_max_loss_ratio) if args.ttt_max_loss_ratio is not None else None),
            max_loss_increase=(
                float(args.ttt_max_loss_increase) if args.ttt_max_loss_increase is not None else None
            ),
            update_filter=str(args.ttt_update_filter),
            include=list(args.ttt_include) if args.ttt_include else None,
            exclude=list(args.ttt_exclude) if args.ttt_exclude else None,
            max_batches=int(args.ttt_max_batches),
            seed=args.ttt_seed,
            log_out=args.ttt_log_out,
            mim_mask_prob=float(args.ttt_mask_prob),
            mim_patch_size=int(args.ttt_patch_size),
            mim_mask_value=float(args.ttt_mask_value),
            cotta_ema_momentum=float(args.ttt_cotta_ema_momentum),
            cotta_augmentations=tuple(args.ttt_cotta_augmentations or ["identity", "hflip"]),
            cotta_aggregation=str(args.ttt_cotta_aggregation),
            cotta_restore_prob=float(args.ttt_cotta_restore_prob),
            cotta_restore_interval=int(args.ttt_cotta_restore_interval),
            eata_conf_min=float(args.ttt_eata_conf_min),
            eata_entropy_min=float(args.ttt_eata_entropy_min),
            eata_entropy_max=float(args.ttt_eata_entropy_max),
            eata_min_valid_dets=int(args.ttt_eata_min_valid_dets),
            eata_anchor_lambda=float(args.ttt_eata_anchor_lambda),
            eata_selected_ratio_min=float(args.ttt_eata_selected_ratio_min),
            eata_max_skip_streak=int(args.ttt_eata_max_skip_streak),
            sar_rho=float(args.ttt_sar_rho),
            sar_adaptive=bool(args.ttt_sar_adaptive),
            sar_first_step_scale=float(args.ttt_sar_first_step_scale),
        )
        if str(args.ttt_reset) == "sample":
            try:
                import torch
            except Exception as exc:  # pragma: no cover
                raise SystemExit(f"TTT failed: {exc}")
            try:
                from yolozu.tta.ttt_mim import select_parameters
            except Exception as exc:  # pragma: no cover
                raise SystemExit(f"TTT failed: {exc}")

            model = adapter.get_model()
            params = select_parameters(
                model,
                update_filter=str(ttt_config.update_filter),
                include=ttt_config.include,
                exclude=ttt_config.exclude,
            )
            if not params:
                raise SystemExit("TTT failed: no parameters selected for TTT")
            with torch.no_grad():
                base_snapshot = [(p, p.detach().clone()) for p in params]
                base_buffers = []
                for name, buffer in model.named_buffers():
                    if buffer is None:
                        continue
                    name = str(name)
                    if not name.endswith(("running_mean", "running_var", "num_batches_tracked")):
                        continue
                    base_buffers.append((buffer, buffer.detach().clone()))

            def _restore_base():
                with torch.no_grad():
                    for p, value in base_snapshot:
                        p.copy_(value)
                    for buffer, value in base_buffers:
                        buffer.copy_(value)

            predictions = []
            per_sample: list[dict] = []
            max_keep = 20
            for idx, record in enumerate(records):
                _restore_base()
                try:
                    rep = _ttt_or_die([record])
                    pred = adapter.predict([record])
                finally:
                    _restore_base()
                predictions.extend(pred)
                if idx < max_keep:
                    per_sample.append({"index": int(idx), "image": record.get("image"), "report": rep})
            ttt_report = {
                "mode": "sample",
                "samples_total": int(len(records)),
                "samples_kept": int(len(per_sample)),
                "samples_truncated": bool(len(records) > max_keep),
                "per_sample": per_sample,
            }
        else:
            ttt_report = _ttt_or_die(records)
            predictions = adapter.predict(records)
    else:
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

    lora_report = None
    if hasattr(adapter, "get_lora_report"):
        try:
            lora_report = adapter.get_lora_report()
        except Exception:
            lora_report = None

    if args.wrap:
        payload = {
            "predictions": predictions,
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "adapter": args.adapter,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "images": len(records),
                "lora": {
                    "enabled": bool(int(args.lora_r) > 0),
                    "r": int(args.lora_r),
                    "alpha": (float(args.lora_alpha) if args.lora_alpha is not None else None),
                    "dropout": float(args.lora_dropout),
                    "target": str(args.lora_target),
                    "freeze_base": bool(args.lora_freeze_base),
                    "train_bias": str(args.lora_train_bias),
                    "report": lora_report,
                },
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
                    "reset": str(args.ttt_reset),
                    "steps": int(args.ttt_steps),
                    "batch_size": int(args.ttt_batch_size),
                    "lr": float(args.ttt_lr),
                    "stop_on_non_finite": bool(args.ttt_stop_on_non_finite),
                    "rollback_on_stop": bool(args.ttt_rollback_on_stop),
                    "max_grad_norm": (float(args.ttt_max_grad_norm) if args.ttt_max_grad_norm is not None else None),
                    "max_update_norm": (float(args.ttt_max_update_norm) if args.ttt_max_update_norm is not None else None),
                    "max_total_update_norm": (
                        float(args.ttt_max_total_update_norm) if args.ttt_max_total_update_norm is not None else None
                    ),
                    "max_loss_ratio": (float(args.ttt_max_loss_ratio) if args.ttt_max_loss_ratio is not None else None),
                    "max_loss_increase": (
                        float(args.ttt_max_loss_increase) if args.ttt_max_loss_increase is not None else None
                    ),
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
                    "cotta": {
                        "ema_momentum": float(args.ttt_cotta_ema_momentum),
                        "augmentations": list(args.ttt_cotta_augmentations or ["identity", "hflip"]),
                        "aggregation": str(args.ttt_cotta_aggregation),
                        "restore_prob": float(args.ttt_cotta_restore_prob),
                        "restore_interval": int(args.ttt_cotta_restore_interval),
                    },
                    "eata": {
                        "conf_min": float(args.ttt_eata_conf_min),
                        "entropy_min": float(args.ttt_eata_entropy_min),
                        "entropy_max": float(args.ttt_eata_entropy_max),
                        "min_valid_dets": int(args.ttt_eata_min_valid_dets),
                        "anchor_lambda": float(args.ttt_eata_anchor_lambda),
                        "selected_ratio_min": float(args.ttt_eata_selected_ratio_min),
                        "max_skip_streak": int(args.ttt_eata_max_skip_streak),
                    },
                    "sar": {
                        "rho": float(args.ttt_sar_rho),
                        "adaptive": bool(args.ttt_sar_adaptive),
                        "first_step_scale": float(args.ttt_sar_first_step_scale),
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
                    "reset": str(args.ttt_reset),
                    "steps": int(args.ttt_steps),
                    "batch_size": int(args.ttt_batch_size),
                    "lr": float(args.ttt_lr),
                    "stop_on_non_finite": bool(args.ttt_stop_on_non_finite),
                    "rollback_on_stop": bool(args.ttt_rollback_on_stop),
                    "max_grad_norm": (float(args.ttt_max_grad_norm) if args.ttt_max_grad_norm is not None else None),
                    "max_update_norm": (float(args.ttt_max_update_norm) if args.ttt_max_update_norm is not None else None),
                    "max_total_update_norm": (
                        float(args.ttt_max_total_update_norm) if args.ttt_max_total_update_norm is not None else None
                    ),
                    "max_loss_ratio": (float(args.ttt_max_loss_ratio) if args.ttt_max_loss_ratio is not None else None),
                    "max_loss_increase": (
                        float(args.ttt_max_loss_increase) if args.ttt_max_loss_increase is not None else None
                    ),
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
                "cotta": {
                    "ema_momentum": float(args.ttt_cotta_ema_momentum),
                    "augmentations": list(args.ttt_cotta_augmentations or ["identity", "hflip"]),
                    "aggregation": str(args.ttt_cotta_aggregation),
                    "restore_prob": float(args.ttt_cotta_restore_prob),
                    "restore_interval": int(args.ttt_cotta_restore_interval),
                },
                "eata": {
                    "conf_min": float(args.ttt_eata_conf_min),
                    "entropy_min": float(args.ttt_eata_entropy_min),
                    "entropy_max": float(args.ttt_eata_entropy_max),
                    "min_valid_dets": int(args.ttt_eata_min_valid_dets),
                    "anchor_lambda": float(args.ttt_eata_anchor_lambda),
                    "selected_ratio_min": float(args.ttt_eata_selected_ratio_min),
                    "max_skip_streak": int(args.ttt_eata_max_skip_streak),
                },
                "sar": {
                    "rho": float(args.ttt_sar_rho),
                    "adaptive": bool(args.ttt_sar_adaptive),
                    "first_step_scale": float(args.ttt_sar_first_step_scale),
                },
                "report": ttt_report,
            },
        }
        log_path.write_text(json.dumps(log_payload, indent=2, sort_keys=True))
        print(log_path)


if __name__ == "__main__":
    main()
