from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolozu import __version__

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


def _cmd_train(config_path: Path) -> int:
    try:
        from rtdetr_pose.tools.train_minimal import main as train_main
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "yolozu train is currently supported from a source checkout (e.g. `pip install -e .`) "
            "because it depends on in-repo trainer scaffolding under rtdetr_pose/tools."
        ) from exc

    return int(train_main(["--config", str(config_path)]))


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
    from yolozu.export import DEFAULT_PREDICTIONS_PATH, export_dummy_predictions, write_predictions_json

    backend = str(getattr(args, "backend", "dummy"))
    if backend != "dummy":
        raise SystemExit("only --backend dummy is currently supported in pip installs")

    dataset = str(args.dataset)
    if not dataset:
        raise SystemExit("--dataset is required")

    try:
        payload, _run = export_dummy_predictions(
            dataset_root=dataset,
            split=str(args.split) if args.split else None,
            max_images=int(args.max_images) if args.max_images is not None else None,
            score=float(args.score),
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    output = str(args.output) if args.output else DEFAULT_PREDICTIONS_PATH
    out_path = write_predictions_json(output=output, payload=payload, force=bool(args.force))
    print(str(out_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yolozu")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Print environment diagnostics as JSON.")
    doctor.add_argument("--output", default="reports/doctor.json", help="Output JSON path (use - for stdout).")

    export = sub.add_parser("export", help="Export predictions.json artifacts.")
    export.add_argument("--backend", choices=("dummy",), default="dummy", help="Inference backend (default: dummy).")
    export.add_argument("--dataset", default="data/coco128", help="YOLO-format dataset root.")
    export.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    export.add_argument("--max-images", type=int, default=50, help="Optional cap for number of images.")
    export.add_argument("--score", type=float, default=0.9, help="Dummy detection score (default: 0.9).")
    export.add_argument(
        "--output",
        default=None,
        help="Predictions JSON output path (default: reports/predictions.json).",
    )
    export.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")

    train = sub.add_parser("train", help="Run training using a YAML/JSON config.")
    train.add_argument("config", type=str, help="Path to train_setting.yaml")

    test = sub.add_parser("test", help="Run scenario tests using a YAML/JSON config.")
    test.add_argument("config", type=str, help="Path to test_setting.yaml")

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
        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"config not found: {config_path}")
        return _cmd_train(config_path)
    if args.command == "test":
        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"config not found: {config_path}")
        return _cmd_test(config_path)
    if args.command == "doctor":
        return _cmd_doctor(str(args.output))
    if args.command == "export":
        return _cmd_export(args)
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
            from yolozu.demos.continual import run_continual_demo

            out = run_continual_demo(
                output=args.output,
                seed=int(args.seed),
                device=str(args.device),
                method=str(args.method),
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
                    "continual demo: "
                    f"accA {a.get('acc_a'):.3f}→{b.get('acc_a'):.3f} "
                    f"accB {a.get('acc_b'):.3f}→{b.get('acc_b'):.3f} "
                    f"forget={forgetting:.3f} gain={gain:.3f}"
                )
            except Exception:
                pass
            print(str(out))
            return 0

    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
