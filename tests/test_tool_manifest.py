import subprocess
import sys
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


if __name__ == "__main__":
    unittest.main()
