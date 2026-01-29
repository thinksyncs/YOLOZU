import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="reports/eval_suite.json", help="Suite output JSON from tools/eval_suite.py")
    parser.add_argument("--targets", default="baselines/yolo26_targets.json", help="Targets JSON path")
    parser.add_argument("--key", default="map50_95", help="Metric key to compare (default: map50_95)")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Require model >= target + min_delta")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    suite = json.loads((repo_root / args.suite).read_text())
    targets_doc = json.loads((repo_root / args.targets).read_text())
    targets = targets_doc.get("targets", {})

    failures = []
    rows = []
    for result in suite.get("results", []):
        name = result.get("name", "")
        metrics = (result.get("metrics") or {})
        value = metrics.get(args.key)

        target = None
        for k, v in targets.items():
            if k in name:
                target = v
                break

        rows.append({"name": name, "value": value, "target": target})

        if target is None:
            failures.append(f"{name}: target missing (set baselines/yolo26_targets.json)")
            continue
        if value is None:
            failures.append(f"{name}: metric missing ({args.key})")
            continue
        if float(value) < float(target) + float(args.min_delta):
            failures.append(f"{name}: {args.key}={value:.4f} < target={target:.4f} (+{args.min_delta})")

    report = {
        "suite": args.suite,
        "targets": args.targets,
        "key": args.key,
        "min_delta": args.min_delta,
        "rows": rows,
        "ok": len(failures) == 0,
        "failures": failures,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

