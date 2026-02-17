import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestImportOptionalDeps(unittest.TestCase):
    def _run(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "yolozu", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )

    def test_import_config_mmdet_missing_dep_is_friendly(self):
        if importlib.util.find_spec("mmengine") is not None:
            self.skipTest("mmengine installed; missing-dep behavior not applicable")
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            cfg = root / "mmdet_config.py"
            cfg.write_text("model = dict()\n", encoding="utf-8")
            out = root / "out.json"
            proc = self._run(
                ["import", "config", "--from", "mmdet", "--config", str(cfg), "--output", str(out), "--force"],
                cwd=repo_root,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("mmengine", proc.stderr.lower())

    def test_import_config_detectron2_missing_dep_is_friendly(self):
        if importlib.util.find_spec("detectron2") is not None:
            self.skipTest("detectron2 installed; missing-dep behavior not applicable")
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            cfg = root / "d2.yaml"
            cfg.write_text("MODEL: {}\n", encoding="utf-8")
            out = root / "out.json"
            proc = self._run(
                ["import", "config", "--from", "detectron2", "--config", str(cfg), "--output", str(out), "--force"],
                cwd=repo_root,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("detectron2", proc.stderr.lower())

    def test_import_config_yolox_missing_dep_is_friendly(self):
        if importlib.util.find_spec("yolox") is not None:
            self.skipTest("yolox installed; missing-dep behavior not applicable")
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            exp = root / "exp.py"
            exp.write_text("import yolox\n", encoding="utf-8")
            out = root / "out.json"
            proc = self._run(
                ["import", "config", "--from", "yolox", "--config", str(exp), "--output", str(out), "--force"],
                cwd=repo_root,
            )
            self.assertNotEqual(proc.returncode, 0)
            # Message includes the original ModuleNotFoundError string.
            self.assertIn("yolox", (proc.stdout + proc.stderr).lower())


if __name__ == "__main__":
    unittest.main()

