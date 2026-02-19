import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.predictions import validate_predictions_payload  # noqa: E402


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

    raw = json.loads(Path(args.predictions).read_text())
    result = validate_predictions_payload(raw, strict=args.strict)

    if isinstance(raw, dict) and "predictions" in raw:
        entries = raw.get("predictions")
        entry_count = len(entries) if isinstance(entries, list) else 0
    else:
        entry_count = len(raw) if isinstance(raw, list) else len(raw or {}) if isinstance(raw, dict) else 0

    if result.warnings:
        for w in result.warnings:
            print(f"WARN: {w}")

    print(f"OK: {entry_count} image entries")


if __name__ == "__main__":
    main()
