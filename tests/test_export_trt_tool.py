import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportTRTTool(unittest.TestCase):
    def test_dry_run_skip_onnx_writes_meta(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_trt.py"
        self.assertTrue(script.is_file(), "missing tools/export_trt.py")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            onnx_path = root / "model.onnx"
            onnx_path.write_bytes(b"dummy")
            engine_path = root / "model_fp16.plan"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--skip-onnx",
                    "--onnx",
                    str(onnx_path),
                    "--engine",
                    str(engine_path),
                    "--precision",
                    "fp16",
                    "--dry-run",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"export_trt.py failed:\n{proc.stdout}\n{proc.stderr}")

            onnx_meta = json.loads((root / "model.onnx.meta.json").read_text())
            self.assertIn("run_record", onnx_meta)
            self.assertEqual(Path(onnx_meta["onnx"]), onnx_path)
            self.assertIn("report", onnx_meta)
            self.assertFalse(bool(onnx_meta["report"].get("exported", True)))

            engine_meta = json.loads((root / "model_fp16.plan.meta.json").read_text())
            self.assertEqual(engine_meta["precision"], "fp16")
            self.assertIn("command_str", engine_meta)
            self.assertIn("nvidia", engine_meta)
            self.assertIn("tensorrt", engine_meta)


if __name__ == "__main__":
    unittest.main()

