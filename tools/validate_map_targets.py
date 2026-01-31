import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.map_targets import load_map_targets_doc


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="baselines/yolo26_targets.json", help="Targets JSON path")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    doc = load_map_targets_doc(repo_root / args.targets)
    print(json.dumps({"ok": True, "targets": args.targets, "metric_key": doc.get("metric_key")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

