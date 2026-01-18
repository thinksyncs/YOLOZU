import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rtdetr_pose.dataset import build_manifest
from rtdetr_pose.validator import validate_manifest


def main():
    dataset_root = repo_root / "data" / "coco128"
    if not dataset_root.exists():
        dataset_root = repo_root.parent / "data" / "coco128"
    manifest = build_manifest(dataset_root)
    validate_manifest(manifest, strict=False, check_content=True)
    summary = {"images": len(manifest["images"])}
    if "stats" in manifest:
        summary["stats"] = manifest["stats"]
    output_dir = repo_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_audit.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
