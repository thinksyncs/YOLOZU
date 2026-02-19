import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.segmentation_predictions import (  # noqa: E402
    normalize_segmentation_predictions_json,
    validate_segmentation_predictions_payload,
)


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("predictions", help="Path to segmentation predictions JSON")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    raw = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    entries = normalize_segmentation_predictions_json(raw)
    result = validate_segmentation_predictions_payload(raw)

    if result.warnings:
        for w in result.warnings:
            print(f"WARN: {w}")

    print(f"OK: {len(entries)} entries")


if __name__ == "__main__":
    main()

