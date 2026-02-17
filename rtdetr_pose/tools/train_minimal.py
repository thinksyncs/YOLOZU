import sys
import importlib
from pathlib import Path

# Source-checkout wrapper.
#
# The implementation lives in `rtdetr_pose/train_minimal.py` so it can be imported
# by the packaged CLI (and bundled by tools like PyInstaller).

repo_root = Path(__file__).resolve().parents[1]
# Ensure the package-root path (repo_root) wins over the workspace root. Otherwise
# Python may resolve `rtdetr_pose` as a namespace package rooted at repo_root.parent,
# which does not contain the actual `rtdetr_pose/train_minimal.py` module.
sys.path.insert(0, str(repo_root.parent))
sys.path.insert(0, str(repo_root))

# If this wrapper is loaded from a source checkout, tests may have already
# imported `rtdetr_pose` as a namespace package (rooted at repo_root.parent).
# Clear it so `import_module("rtdetr_pose.train_minimal")` resolves to the real
# package under repo_root.
sys.modules.pop("rtdetr_pose", None)
for key in list(sys.modules.keys()):
    if key.startswith("rtdetr_pose."):
        sys.modules.pop(key, None)

_impl = importlib.import_module("rtdetr_pose.train_minimal")
__all__ = [name for name in dir(_impl) if not (name.startswith("__") and name.endswith("__"))]


def __getattr__(name):  # pragma: no cover
    return getattr(_impl, name)


def __dir__():  # pragma: no cover
    return sorted(set(globals().keys()) | set(__all__))


if __name__ == "__main__":
    raise SystemExit(_impl.main())
