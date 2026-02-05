import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.splits import deterministic_split_paths  # noqa: E402


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True, help="Directory containing images to split.")
    p.add_argument("--glob", default="*.jpg", help="Glob pattern for images (default: *.jpg).")
    p.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction (default: 0.1).")
    p.add_argument("--seed", type=int, default=0, help="Split seed (default: 0).")
    p.add_argument("--output", default="reports/split.json", help="Output JSON path.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    images_dir = Path(args.images_dir)
    paths = sorted(images_dir.glob(args.glob))
    split = deterministic_split_paths(paths, val_fraction=args.val_fraction, seed=args.seed)

    out = {
        "images_dir": str(images_dir),
        "glob": args.glob,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "train": split.train,
        "val": split.val,
    }

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()

