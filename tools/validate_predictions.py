import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.predictions import load_predictions_entries, validate_predictions_entries  # noqa: E402


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="Path to predictions JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require numeric types and full per-detection schema checks.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    entries = load_predictions_entries(args.predictions)
    result = validate_predictions_entries(entries, strict=args.strict)

    if result.warnings:
        for w in result.warnings:
            print(f"WARN: {w}")

    print(f"OK: {len(entries)} image entries")


if __name__ == "__main__":
    main()
