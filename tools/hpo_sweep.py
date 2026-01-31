import argparse
import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slugify(value: str) -> str:
    value = value.replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return value.strip("-_")


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _make_run_id(params: dict, keys: list[str]) -> str:
    parts = [f"{k}-{_slugify(_format_value(params[k]))}" for k in keys]
    base = "__".join([p for p in parts if p])
    if not base:
        return "run"
    if len(base) <= 80:
        return base
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{base[:60].rstrip('-_')}__{digest}"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _extract_key_path(data: dict, key_path: str):
    cur = data
    for key in key_path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _build_param_list(config: dict) -> list[dict]:
    if "param_list" in config:
        if not isinstance(config["param_list"], list):
            raise SystemExit("param_list must be a list of param dicts")
        return [dict(item) for item in config["param_list"]]
    grid = config.get("param_grid") or {}
    if not isinstance(grid, dict) or not grid:
        raise SystemExit("param_grid is required (or provide param_list)")
    keys = list(grid.keys())
    values = []
    for key in keys:
        vals = grid[key]
        if not isinstance(vals, list) or not vals:
            raise SystemExit(f"param_grid[{key}] must be a non-empty list")
        values.append(vals)
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _load_existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    results = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        run_id = obj.get("run_id")
        if run_id:
            results[run_id] = obj
    return results


def _write_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in keys})


def _write_md(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    lines = []
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
    for row in rows:
        vals = [str(row.get(k, "")) for k in keys]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def _result_row(result: dict) -> dict:
    row = {
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "exit_code": result.get("exit_code"),
        "elapsed_sec": result.get("elapsed_sec"),
    }
    params = result.get("params") or {}
    metrics = result.get("metrics") or {}
    row.update({f"params.{k}": v for k, v in params.items()})
    row.update({f"metrics.{k}": v for k, v in metrics.items()})
    return row


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to sweep config JSON.")
    p.add_argument("--resume", action="store_true", help="Skip runs already present in results jsonl.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p.add_argument("--max-runs", type=int, default=None, help="Optional cap for number of runs.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    config = _load_json(cfg_path)

    base_cmd = config.get("base_cmd")
    if not base_cmd:
        raise SystemExit("config requires base_cmd")

    shell = bool(config.get("shell", True))
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (config.get("env") or {}).items()})

    run_dir_template = config.get("run_dir", "runs/hpo/{run_id}")
    result_jsonl = Path(config.get("result_jsonl", "reports/hpo_sweep.jsonl"))
    result_csv = Path(config.get("result_csv", "reports/hpo_sweep.csv"))
    result_md = Path(config.get("result_md", "reports/hpo_sweep.md"))

    metrics_cfg = config.get("metrics") or {}
    metrics_path_template = metrics_cfg.get("path")
    metrics_keys = metrics_cfg.get("keys") or []

    param_list = _build_param_list(config)
    param_order = config.get("param_order") or sorted(param_list[0].keys())
    existing = _load_existing_results(result_jsonl) if args.resume else {}

    results = list(existing.values()) if existing else []
    run_count = 0

    for params in param_list:
        run_id = _make_run_id(params, param_order)
        if args.resume and run_id in existing:
            continue

        run_dir = run_dir_template.format(run_id=run_id, **params)
        cmd = base_cmd.format(run_id=run_id, run_dir=run_dir, **params)

        if args.dry_run:
            print(cmd)
            continue

        start = time.time()
        proc = subprocess.run(cmd if shell else shlex.split(cmd), shell=shell, env=env)
        elapsed = time.time() - start

        status = "ok" if proc.returncode == 0 else "failed"
        metrics = {}
        metrics_missing = []

        if metrics_path_template:
            metrics_path = Path(metrics_path_template.format(run_id=run_id, run_dir=run_dir, **params))
            if metrics_path.exists():
                raw = _load_json(metrics_path)
                if metrics_keys:
                    for key in metrics_keys:
                        value = _extract_key_path(raw, key)
                        if value is None:
                            metrics_missing.append(key)
                        else:
                            metrics[key] = value
                else:
                    if isinstance(raw, dict):
                        metrics = raw
            else:
                status = "missing_metrics" if status == "ok" else status
                metrics_missing.append("metrics.path")

        result = {
            "timestamp": _now_utc(),
            "run_id": run_id,
            "params": params,
            "run_dir": run_dir,
            "command": cmd,
            "shell": shell,
            "status": status,
            "exit_code": proc.returncode,
            "elapsed_sec": round(elapsed, 3),
            "metrics": metrics,
            "metrics_missing": metrics_missing,
        }
        _write_jsonl(result_jsonl, result)
        results.append(result)

        run_count += 1
        if args.max_runs is not None and run_count >= args.max_runs:
            break

    rows = [_result_row(r) for r in results]
    if rows:
        _write_csv(result_csv, rows)
        _write_md(result_md, rows)


if __name__ == "__main__":
    main()
