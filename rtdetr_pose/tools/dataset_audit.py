import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rtdetr_pose.dataset import build_manifest
from rtdetr_pose.validator import validate_sample


def main():
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
            validate_sample(sample, strict=False, check_content=True, check_ranges=True)
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

            if len(issues["examples"]) < 25:
                issues["examples"].append(
                    {
                        "index": idx,
                        "image_path": str(sample.get("image_path")),
                        "error": msg,
                    }
                )

    summary["issues"] = issues
    output_dir = repo_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_audit.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
