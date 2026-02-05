import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.scenario_suite import build_report


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        default="reports/scenario_suite.json",
        help="Where to write scenario suite report JSON.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = build_report()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
