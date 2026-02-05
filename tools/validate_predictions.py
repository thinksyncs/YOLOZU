import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.predictions import normalize_predictions_payload, validate_predictions_entries, validate_wrapped_meta  # noqa: E402


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
    entries, meta = normalize_predictions_payload(raw)
    if meta is not None:
        validate_wrapped_meta(meta)
    result = validate_predictions_entries(entries, strict=args.strict)

    if result.warnings:
        for w in result.warnings:
            print(f"WARN: {w}")

    print(f"OK: {len(entries)} image entries")


if __name__ == "__main__":
    main()
