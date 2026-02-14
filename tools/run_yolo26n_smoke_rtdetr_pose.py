import shlex
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    python = sys.executable
    args = sys.argv[1:] if argv is None else argv
    cmd = [python, "tools/run_yolo26_smoke_rtdetr_pose.py", *args, "--buckets", "n"]
    print(shlex.join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
