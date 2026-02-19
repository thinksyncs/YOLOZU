import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGoldenCompatibilityTool(unittest.TestCase):
    def test_golden_compatibility_tool_passes_repo_manifest(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_golden_compatibility.py"
        self.assertTrue(script.is_file())

        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"check_golden_compatibility.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

        payload = json.loads((proc.stdout or "").strip())
        self.assertTrue(payload.get("ok"))
        self.assertGreaterEqual(int(payload.get("assets_checked") or 0), 4)

    def test_golden_compatibility_tool_fails_on_hash_mismatch(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_golden_compatibility.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            manifest_src = repo_root / "baselines" / "golden" / "v1" / "manifest.json"
            manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
            manifest["assets"][0]["expected_sha256"] = "0" * 64
            manifest_path = root / "manifest_bad.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, str(script), "--manifest", str(manifest_path.relative_to(repo_root))],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("hash mismatch", (proc.stdout or "") + (proc.stderr or ""))


if __name__ == "__main__":
    unittest.main()
