import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.scenarios_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

