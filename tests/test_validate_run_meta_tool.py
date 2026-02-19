import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestValidateRunMetaTool(unittest.TestCase):
    def test_validate_run_meta_tool_accepts_complete_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "validate_run_meta.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            run_meta = root / "run_meta.json"
            run_meta.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "git": {"sha": "abc123", "dirty": False},
                        "dependency_lock": {
                            "python_version": "3.11.0",
                            "package_count": 1,
                            "package_set_sha256": "a" * 64,
                            "requirements_files_sha256": {},
                        },
                        "preprocess": {"image_size": 64, "multiscale": False, "scale_min": None, "scale_max": None},
                        "hardware": {"host": {}, "accelerator": {}},
                        "runtime": {
                            "python_version": "3.11.0",
                            "platform": "test",
                            "python_executable": "/usr/bin/python3",
                        },
                        "command": {
                            "argv": ["--config", "x.yaml"],
                            "command": ["python3", "train.py", "--config", "x.yaml"],
                            "command_str": "python3 train.py --config x.yaml",
                            "python_executable": "/usr/bin/python3",
                            "cwd": "/tmp",
                        },
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [sys.executable, str(script), str(run_meta.relative_to(repo_root))],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"validate_run_meta.py failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertIn("OK:", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
