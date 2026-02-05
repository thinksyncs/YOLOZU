import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.metrics_report import build_report, write_json  # noqa: E402


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Output directory for metrics.json")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--wd", type=float, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args(argv)


def _score(lr: float, wd: float, imgsz: int) -> float:
    base = 0.5
    base -= abs(lr - 0.01) * 10.0
    base -= abs(wd - 0.0005) * 100.0
    base -= abs(imgsz - 640) / 640.0 * 0.05
    return max(0.0, min(1.0, base))


def main(argv=None):
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    score = _score(args.lr, args.wd, args.imgsz)
    losses = {"total": round(1.0 - score, 6)}
    metrics = {"map50_95": round(score, 6)}
    report = build_report(losses=losses, metrics=metrics, meta={"lr": args.lr, "wd": args.wd, "imgsz": args.imgsz})

    write_json(run_dir / "metrics.json", report)


if __name__ == "__main__":
    main()
