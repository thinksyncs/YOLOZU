import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.benchmark import run_benchmark
from yolozu.scenario_suite import build_report


def main():
    report = build_report()
    fps = run_benchmark(iterations=100, sleep_s=0.0)
    baseline = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fps": fps,
        "scenario_report": report,
    }
    output_dir = repo_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline.json"
    output_path.write_text(json.dumps(baseline, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
