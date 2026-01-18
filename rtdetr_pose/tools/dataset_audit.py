import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rtdetr_pose.dataset import build_manifest
from rtdetr_pose.validator import validate_sample


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit dataset manifest + sidecar metadata")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root directory (default: repo_root/data/coco128 or ../data/coco128)",
    )
    parser.add_argument("--strict", action="store_true", help="Require mask/depth/pose/intrinsics")
    parser.add_argument(
        "--check-content",
        action="store_true",
        help="Perform content checks (mask/depth consistency, CAD projection sanity)",
    )
    parser.add_argument(
        "--check-ranges",
        action="store_true",
        help="Check range/unit conventions (mask binary, depth finite/non-negative)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=25,
        help="Max number of failing samples to include in JSON report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: repo_root/reports/dataset_audit.json)",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with code 2 if any issues are found",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = repo_root / "data" / "coco128"
        if not dataset_root.exists():
            dataset_root = repo_root.parent / "data" / "coco128"

    manifest = build_manifest(dataset_root)
    summary = {"images": len(manifest["images"])}
    if "stats" in manifest:
        summary["stats"] = manifest["stats"]

    issues = {
        "total": 0,
        "by_type": {
            "mask_nonbinary": 0,
            "mask_nonfinite": 0,
            "mask_float_range": 0,
            "depth_negative": 0,
            "depth_nonfinite": 0,
            "other": 0,
        },
        "examples": [],
    }

    for idx, sample in enumerate(manifest.get("images", [])):
        try:
            validate_sample(
                sample,
                strict=bool(args.strict),
                check_content=bool(args.check_content),
                check_ranges=bool(args.check_ranges),
            )
        except Exception as exc:
            msg = str(exc)
            issues["total"] += 1
            if "mask must be binary" in msg:
                issues["by_type"]["mask_nonbinary"] += 1
            elif "mask must be finite" in msg:
                issues["by_type"]["mask_nonfinite"] += 1
            elif "mask float values must be in [0,1]" in msg:
                issues["by_type"]["mask_float_range"] += 1
            elif "depth must be non-negative" in msg:
                issues["by_type"]["depth_negative"] += 1
            elif "depth must be finite" in msg:
                issues["by_type"]["depth_nonfinite"] += 1
            else:
                issues["by_type"]["other"] += 1

            if len(issues["examples"]) < int(args.max_examples):
                issues["examples"].append(
                    {
                        "index": idx,
                        "image_path": str(sample.get("image_path")),
                        "error": msg,
                    }
                )

    summary["issues"] = issues
    if args.output is None:
        output_dir = repo_root / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dataset_audit.json"
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Dataset root: {dataset_root}")
    print(f"Images: {summary['images']}")
    if "stats" in summary:
        print("Availability stats present: yes")
    print(f"Issues: {issues['total']}")
    if issues["total"]:
        print(f"Issue types: {issues['by_type']}")
    print(f"Report: {output_path}")

    if args.fail_on_issues and issues["total"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
