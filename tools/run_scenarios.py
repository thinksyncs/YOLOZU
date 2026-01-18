import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import DummyAdapter
from yolozu.dataset import build_manifest
from yolozu.runner import run_adapter


def main():
    dataset_root = repo_root / "data" / "coco128"
    manifest = build_manifest(dataset_root)
    adapter = DummyAdapter()
    result = run_adapter(adapter, manifest["images"])
    output_dir = repo_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scenario_run.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
