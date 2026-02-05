import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.predictions import normalize_predictions_json  # noqa: E402
from yolozu.predictions_transform import load_classes_json, normalize_class_ids  # noqa: E402


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input predictions JSON.")
    p.add_argument("--output", required=True, help="Output predictions JSON.")
    p.add_argument(
        "--classes",
        default=None,
        help="Path to labels/<split>/classes.json (for category_id -> class_id mapping).",
    )
    p.add_argument(
        "--assume-class-id-is-category-id",
        action="store_true",
        help="Treat detection.class_id as a COCO category_id and remap to contiguous class_id.",
    )
    p.add_argument(
        "--wrap",
        action="store_true",
        help="Write as {predictions:[...], meta:{...}} regardless of input shape.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    input_path = repo_root / args.input
    raw = json.loads(input_path.read_text())
    entries = normalize_predictions_json(raw)

    classes_json = load_classes_json(repo_root / args.classes) if args.classes else None
    transformed = normalize_class_ids(
        entries,
        classes_json=classes_json,
        assume_class_id_is_category_id=bool(args.assume_class_id_is_category_id),
    )

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.wrap:
        payload = {
            "predictions": transformed.entries,
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": str(args.input),
                "classes": str(args.classes) if args.classes else None,
                "assume_class_id_is_category_id": bool(args.assume_class_id_is_category_id),
                "warnings": transformed.warnings,
            },
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    else:
        # Preserve original top-level wrapper if present.
        if isinstance(raw, dict) and "predictions" in raw:
            raw = dict(raw)
            raw["predictions"] = transformed.entries
            if "meta" in raw and isinstance(raw["meta"], dict):
                raw["meta"] = dict(raw["meta"])
                raw["meta"]["normalized_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                raw["meta"]["warnings"] = transformed.warnings
            output_path.write_text(json.dumps(raw, indent=2, sort_keys=True))
        else:
            output_path.write_text(json.dumps(transformed.entries, indent=2, sort_keys=True))

    print(output_path)


if __name__ == "__main__":
    main()

