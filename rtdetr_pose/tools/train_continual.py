#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "rtdetr_pose"))
sys.path.insert(0, str(repo_root))

from rtdetr_pose.dataset import build_manifest  # noqa: E402

from yolozu.replay_buffer import ReplayBuffer  # noqa: E402
from yolozu.run_record import build_run_record  # noqa: E402


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_json(obj: Any) -> str:
    import hashlib

    data = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required for YAML configs; install requirements.txt") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"config must be a mapping at top-level: {path}")
    return dict(data)


def _resolve(path_str: str | None, *, base: Path) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _task_dir(run_dir: Path, idx: int, name: str) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(name).strip()) or f"task{idx:02d}"
    return run_dir / f"task{int(idx):02d}_{safe}"


def _train_minimal_cmd(*, python: str, args: list[str]) -> list[str]:
    return [python, str(repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"), *args]


def _subprocess_or_die(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _as_cli_args(d: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in d.items():
        if value is None:
            continue
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            out.append(flag)
            out.extend([str(v) for v in value])
            continue
        out.append(flag)
        out.append(str(value))
    return out


@dataclass(frozen=True)
class _TaskCfg:
    name: str
    dataset_root: Path
    train_split: str
    val_split: str


def _parse_tasks(cfg: dict[str, Any]) -> list[_TaskCfg]:
    tasks_raw = cfg.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise SystemExit("config.tasks must be a non-empty list")
    tasks: list[_TaskCfg] = []
    for i, item in enumerate(tasks_raw):
        if not isinstance(item, dict):
            raise SystemExit(f"task[{i}] must be a mapping")
        name = str(item.get("name") or f"task{i:02d}")
        root_s = item.get("dataset_root")
        if not isinstance(root_s, str) or not root_s:
            raise SystemExit(f"task[{i}].dataset_root is required")
        dataset_root = _resolve(root_s, base=repo_root)
        assert dataset_root is not None
        train_split = str(item.get("train_split") or item.get("split") or "train2017")
        val_split = str(item.get("val_split") or "val2017")
        tasks.append(_TaskCfg(name=name, dataset_root=dataset_root, train_split=train_split, val_split=val_split))
    return tasks


def _load_records(dataset_root: Path, *, split: str) -> list[dict[str, Any]]:
    manifest = build_manifest(dataset_root, split=str(split))
    records = manifest.get("images") or []
    records = [r for r in records if isinstance(r, dict)]
    records.sort(key=lambda r: str(r.get("image_path", "")))
    return records


def _to_abs_paths(records: list[dict[str, Any]], *, base: Path) -> list[dict[str, Any]]:
    def _abs(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            p = Path(value)
            if not p.is_absolute():
                return str((base / p).resolve())
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
            out: list[Any] = []
            for item in value:
                out.append(_abs(item))
            return out
        return value

    out: list[dict[str, Any]] = []
    for rec in records:
        copied = dict(rec)
        for key in ("image_path", "mask_path", "mask", "depth_path", "depth", "cad_points"):
            if key in copied:
                copied[key] = _abs(copied.get(key))
        out.append(copied)
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continual fine-tuning runner for rtdetr_pose (domain-incremental).")
    p.add_argument("--config", required=True, help="YAML/JSON continual learning config.")
    p.add_argument("--run-dir", default=None, help="Output directory for this continual run.")
    p.add_argument("--replay-size", type=int, default=None, help="Override continual.replay_size (0 disables replay).")
    p.add_argument(
        "--replay-fraction",
        type=float,
        default=None,
        help="Override continual.replay_fraction (replay_k = fraction * train_records; default: use all buffer items).",
    )
    p.add_argument(
        "--replay-per-task-cap",
        type=int,
        default=None,
        help="Override continual.replay_per_task_cap (cap replay samples per past task; default: no cap).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    cfg = _load_config(cfg_path)
    schema_version = int(cfg.get("schema_version") or 1)
    if schema_version != 1:
        raise SystemExit(f"unsupported schema_version: {schema_version}")

    model_config = cfg.get("model_config")
    if not isinstance(model_config, str) or not model_config:
        raise SystemExit("config.model_config is required (rtdetr_pose JSON config)")
    model_config_path = _resolve(model_config, base=repo_root)
    assert model_config_path is not None

    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    continual_cfg = cfg.get("continual") if isinstance(cfg.get("continual"), dict) else {}

    seed = int(continual_cfg.get("seed") or train_cfg.get("seed") or 0)
    replay_cfg = continual_cfg.get("replay_size")
    replay_size = int(replay_cfg) if replay_cfg is not None else 50
    if args.replay_size is not None:
        replay_size = int(args.replay_size)
    replay_size = max(0, int(replay_size))

    replay_fraction_cfg = continual_cfg.get("replay_fraction")
    replay_fraction = float(replay_fraction_cfg) if replay_fraction_cfg is not None else None
    if args.replay_fraction is not None:
        replay_fraction = float(args.replay_fraction)
    if replay_fraction is not None and replay_fraction < 0.0:
        raise SystemExit("continual.replay_fraction must be >= 0")

    replay_per_task_cap_cfg = continual_cfg.get("replay_per_task_cap")
    replay_per_task_cap = int(replay_per_task_cap_cfg) if replay_per_task_cap_cfg is not None else None
    if args.replay_per_task_cap is not None:
        replay_per_task_cap = int(args.replay_per_task_cap)
    if replay_per_task_cap is not None and replay_per_task_cap < 0:
        raise SystemExit("continual.replay_per_task_cap must be >= 0")

    distill_cfg = continual_cfg.get("distill") if isinstance(continual_cfg.get("distill"), dict) else {}
    distill_enabled = bool(distill_cfg.get("enabled", True))

    lora_cfg = continual_cfg.get("lora") if isinstance(continual_cfg.get("lora"), dict) else {}
    lora_enabled = bool(lora_cfg.get("enabled", False))

    tasks = _parse_tasks(cfg)

    run_dir = _resolve(args.run_dir, base=repo_root) if args.run_dir else None
    if run_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        run_dir = repo_root / "runs" / "continual" / f"{stamp}_rtdetr_pose"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_record = build_run_record(
        repo_root=repo_root,
        argv=(sys.argv[1:] if argv is None else argv),
        args={
            "config": str(cfg_path),
            "run_dir": str(run_dir),
            "replay_size": int(replay_size),
            "replay_fraction": replay_fraction,
            "replay_per_task_cap": replay_per_task_cap,
            "seed": int(seed),
        },
        extra={"timestamp_utc": _now_utc()},
    )

    buffer = ReplayBuffer(capacity=int(replay_size), seed=int(seed))

    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": _now_utc(),
        "config_path": str(cfg_path),
        "config_hash": _sha256_json(cfg),
        "model_config": str(model_config_path),
        "seed": int(seed),
        "replay": {
            "size": int(replay_size),
            "strategy": str(continual_cfg.get("replay_strategy") or "reservoir"),
            "fraction": replay_fraction,
            "per_task_cap": replay_per_task_cap,
            "task_key": "__task",
        },
        "distill": {
            "enabled": bool(distill_enabled),
            "weight": float(distill_cfg.get("weight", 1.0)),
            "temperature": float(distill_cfg.get("temperature", 1.0)),
            "kl": str(distill_cfg.get("kl", "reverse")),
            "keys": str(distill_cfg.get("keys", "logits,bbox")),
        },
        "lora": dict(lora_cfg) if isinstance(lora_cfg, dict) else {},
        "tasks": [],
        "run_record": run_record,
    }

    prev_ckpt: Path | None = None

    for task_idx, task in enumerate(tasks):
        task_out = _task_dir(run_dir, task_idx, task.name)
        task_out.mkdir(parents=True, exist_ok=True)

        train_records = _to_abs_paths(_load_records(task.dataset_root, split=task.train_split), base=repo_root)
        replay_k = None
        if replay_fraction is not None:
            replay_k = max(0, int(round(float(replay_fraction) * float(len(train_records)))))

        sample_kwargs: dict[str, Any] = {}
        if replay_per_task_cap is not None:
            sample_kwargs = {"task_key": "__task", "per_task_cap": int(replay_per_task_cap)}
        buffer_records = buffer.sample(replay_k, **sample_kwargs)

        extra_records_path = None
        if buffer_records:
            extra_records_path = task_out / "replay_records.json"
            extra_records_path.write_text(
                json.dumps(buffer_records, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )

        # Train args forwarded to train_minimal.
        forwarded = dict(train_cfg) if isinstance(train_cfg, dict) else {}
        forwarded.pop("seed", None)  # handled explicitly
        forwarded.pop("dataset_root", None)
        forwarded.pop("split", None)

        train_args: list[str] = [
            "--config",
            str(model_config_path),
            "--dataset-root",
            str(task.dataset_root),
            "--split",
            str(task.train_split),
            "--run-dir",
            str(task_out),
            "--seed",
            str(int(seed)),
        ]

        if extra_records_path is not None:
            train_args.extend(["--extra-records-json", str(extra_records_path)])

        if prev_ckpt is not None:
            train_args.extend(["--resume-from", str(prev_ckpt)])
            if distill_enabled:
                train_args.extend(["--self-distill-from", str(prev_ckpt)])
                train_args.extend(["--self-distill-keys", str(run_meta["distill"]["keys"])])
                train_args.extend(["--self-distill-weight", str(float(run_meta["distill"]["weight"]))])
                train_args.extend(["--self-distill-temperature", str(float(run_meta["distill"]["temperature"]))])
                train_args.extend(["--self-distill-kl", str(run_meta["distill"]["kl"])])

        if lora_enabled:
            train_args.extend(["--lora-r", str(int(lora_cfg.get("r") or 0))])
            if lora_cfg.get("alpha") is not None:
                train_args.extend(["--lora-alpha", str(float(lora_cfg.get("alpha")))])
            train_args.extend(["--lora-dropout", str(float(lora_cfg.get("dropout") or 0.0))])
            train_args.extend(["--lora-target", str(lora_cfg.get("target") or "head")])
            freeze_base = bool(lora_cfg.get("freeze_base", True))
            train_args.append("--lora-freeze-base" if freeze_base else "--no-lora-freeze-base")
            train_args.extend(["--lora-train-bias", str(lora_cfg.get("train_bias") or "none")])

        train_args.extend(_as_cli_args(forwarded))

        cmd = _train_minimal_cmd(python=sys.executable, args=train_args)
        _subprocess_or_die(cmd)

        ckpt_path = task_out / "checkpoint.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"expected checkpoint not found: {ckpt_path}")

        # Update replay buffer AFTER training on this task.
        tagged_train_records = [dict(rec, __task=str(task.name)) for rec in train_records]
        buffer.add_many(tagged_train_records)

        prev_ckpt = ckpt_path

        run_meta["tasks"].append(
            {
                "idx": int(task_idx),
                "name": task.name,
                "dataset_root": str(task.dataset_root),
                "train_split": str(task.train_split),
                "val_split": str(task.val_split),
                "train_records": int(len(train_records)),
                "replay_requested": replay_k,
                "replay_used": int(len(buffer_records)),
                "run_dir": str(task_out),
                "checkpoint": str(ckpt_path),
            }
        )

        (run_dir / "replay_buffer.json").write_text(
            json.dumps(buffer.summary(), indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "continual_run.json").write_text(
            json.dumps(run_meta, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    print(run_dir / "continual_run.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
