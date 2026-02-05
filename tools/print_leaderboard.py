import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.map_targets import load_targets_map  # noqa: E402


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="reports/eval_suite.json", help="Suite JSON from tools/eval_suite.py")
    p.add_argument("--targets", default=None, help="Targets JSON (e.g. baselines/yolo26_targets.json)")
    p.add_argument("--key", default="map50_95", help="Metric key (default: map50_95)")
    p.add_argument("--format", choices=("md", "tsv", "json"), default="md")
    return p.parse_args(argv)


def _load_targets(path: str | None) -> dict[str, float | None]:
    if not path:
        return {}
    return load_targets_map(repo_root / path)


def _match_target(name: str, targets: dict[str, float | None]) -> float | None:
    for k, v in targets.items():
        if k in name:
            return v
    return None


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    suite_path = repo_root / args.suite
    if not suite_path.exists():
        raise SystemExit(f"suite not found: {suite_path} (run tools/eval_suite.py first)")
    suite = json.loads(suite_path.read_text())
    targets = _load_targets(args.targets)

    rows = []
    for res in suite.get("results", []):
        name = str(res.get("name", ""))
        metrics = res.get("metrics") or {}
        value = metrics.get(args.key)
        try:
            value_f = None if value is None else float(value)
        except Exception:
            value_f = None
        target = _match_target(name, targets)
        delta = None if (value_f is None or target is None) else (value_f - target)
        rows.append(
            {
                "name": name,
                "value": value_f,
                "target": target,
                "delta": delta,
                "path": res.get("path"),
            }
        )

    rows.sort(key=lambda r: (-1.0 if r["value"] is None else -r["value"], r["name"]))

    if args.format == "json":
        print(json.dumps({"suite": args.suite, "key": args.key, "rows": rows}, indent=2, sort_keys=True))
        return

    if args.format == "tsv":
        print("\t".join(("name", args.key, "target", "delta", "path")))
        for r in rows:
            v = "" if r["value"] is None else f"{r['value']:.6g}"
            t = "" if r["target"] is None else f"{r['target']:.6g}"
            d = "" if r["delta"] is None else f"{r['delta']:+.6g}"
            print("\t".join((r["name"], v, t, d, str(r.get("path") or ""))))
        return

    # md
    print(f"| name | {args.key} | target | delta |")
    print("|---|---:|---:|---:|")
    for r in rows:
        v = "" if r["value"] is None else f"{r['value']:.4f}"
        t = "" if r["target"] is None else f"{r['target']:.4f}"
        d = "" if r["delta"] is None else f"{r['delta']:+.4f}"
        print(f"| {r['name']} | {v} | {t} | {d} |")


if __name__ == "__main__":
    main()
