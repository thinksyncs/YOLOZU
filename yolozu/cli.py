from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    from rtdetr_pose.tools.train_minimal import main as train_main

    return int(train_main(["--config", str(config_path)]))


def _cmd_test(config_path: Path) -> int:
    from tools.run_scenarios import main as scenarios_main

    cfg = _load_config(config_path)
    args = _build_args_from_config(cfg)
    scenarios_main(args)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yolozu")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run training using a YAML/JSON config.")
    train.add_argument("config", type=str, help="Path to train_setting.yaml")

    test = sub.add_parser("test", help="Run scenario tests using a YAML/JSON config.")
    test.add_argument("config", type=str, help="Path to test_setting.yaml")

    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")

    if args.command == "train":
        return _cmd_train(config_path)
    if args.command == "test":
        return _cmd_test(config_path)

    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
