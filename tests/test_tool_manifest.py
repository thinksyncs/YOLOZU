import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestToolManifest(unittest.TestCase):
    def test_validate_tool_manifest(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "validate_tool_manifest.py"
        self.assertTrue(script.is_file(), "missing tools/validate_tool_manifest.py")

        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"tool manifest validation failed:\n{proc.stdout}\n{proc.stderr}")

    def test_validate_tool_manifest_declarative_mode_fails_missing_fields(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "validate_tool_manifest.py"

        bad_manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "dummy_tool",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "dummy",
                    "platform": {
                        "cpu_ok": True,
                        "gpu_required": False,
                        "macos_ok": True,
                        "linux_ok": True,
                    },
                    "effects": {"writes": []},
                    "examples": [{"description": "x", "command": "python3 tools/validate_tool_manifest.py"}],
                }
            ],
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fp:
            import json

            json.dump(bad_manifest, fp)
            tmp_path = fp.name
        try:
            proc = subprocess.run(
                [sys.executable, str(script), "--manifest", tmp_path, "--require-declarative"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("required in declarative mode", proc.stderr)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_tool_manifest_declarative_mode_accepts_compliant(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "validate_tool_manifest.py"

        ok_manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "dummy_tool_ok",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "dummy",
                    "platform": {
                        "cpu_ok": True,
                        "gpu_required": False,
                        "macos_ok": True,
                        "linux_ok": True,
                    },
                    "inputs": [{"name": "manifest", "kind": "file", "required": False, "flag": "--manifest"}],
                    "effects": {"writes": [], "fixed_writes": []},
                    "outputs": [{"name": "status", "kind": "stdout"}],
                    "examples": [{"description": "x", "command": "python3 tools/validate_tool_manifest.py --help"}],
                }
            ],
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fp:
            import json

            json.dump(ok_manifest, fp)
            tmp_path = fp.name
        try:
            proc = subprocess.run(
                [sys.executable, str(script), "--manifest", tmp_path, "--require-declarative"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
