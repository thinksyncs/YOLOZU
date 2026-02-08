import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestRunRTDETRPoseBackendSuiteCLI(unittest.TestCase):
    def test_dry_run_skip_onnx_writes_artifacts(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_rtdetr_pose_backend_suite.py"
        self.assertTrue(script.is_file(), "missing tools/run_rtdetr_pose_backend_suite.py")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            run_dir = root / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Provide a dummy ONNX so export_trt.py can write engine meta in --dry-run mode.
            (run_dir / "model.onnx").write_bytes(b"dummy")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    "rtdetr_pose/configs/base.json",
                    "--run-dir",
                    str(run_dir),
                    "--skip-onnx",
                    "--dry-run",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"run_rtdetr_pose_backend_suite.py failed:\n{proc.stdout}\n{proc.stderr}")

            pipeline = json.loads((run_dir / "pipeline.json").read_text(encoding="utf-8"))
            self.assertIn("artifacts", pipeline)

            self.assertTrue((run_dir / "model.onnx.meta.json").is_file())
            self.assertTrue((run_dir / "model_fp16.plan.meta.json").is_file())

            report_path = Path(pipeline["artifacts"]["suite_report"])
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("meta", report)
            self.assertIn("system", report.get("meta", {}))


if __name__ == "__main__":
    unittest.main()

