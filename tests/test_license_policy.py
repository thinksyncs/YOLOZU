import subprocess
import unittest
from pathlib import Path


class TestLicensePolicy(unittest.TestCase):
    def test_license_policy_script(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_license_policy.py"
        self.assertTrue(script.is_file(), "missing tools/check_license_policy.py")

        proc = subprocess.run(
            ["python3", str(script)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"license policy check failed:\n{proc.stdout}\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main()

