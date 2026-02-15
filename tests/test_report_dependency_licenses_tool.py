import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestReportDependencyLicensesTool(unittest.TestCase):
    def test_report_dependency_licenses_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "report_dependency_licenses.py"
        self.assertTrue(script.is_file(), "missing tools/report_dependency_licenses.py")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_path = root / "licenses.json"

            proc = subprocess.run(
                [sys.executable, str(script), "--output", str(out_path)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"report_dependency_licenses.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("summary", payload)
            self.assertIn("packages", payload)
            self.assertIsInstance(payload["packages"], list)


if __name__ == "__main__":
    unittest.main()
