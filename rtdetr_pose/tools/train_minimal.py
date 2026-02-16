import sys
import importlib
from pathlib import Path

# Source-checkout wrapper.
#
# The implementation lives in `rtdetr_pose/train_minimal.py` so it can be imported
# by the packaged CLI (and bundled by tools like PyInstaller).

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root.parent))

_impl = importlib.import_module("rtdetr_pose.train_minimal")
__all__ = [name for name in dir(_impl) if not (name.startswith("__") and name.endswith("__"))]


def __getattr__(name):  # pragma: no cover
    return getattr(_impl, name)


def __dir__():  # pragma: no cover
    return sorted(set(globals().keys()) | set(__all__))


if __name__ == "__main__":
    raise SystemExit(_impl.main())
