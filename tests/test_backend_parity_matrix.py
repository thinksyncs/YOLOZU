import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_MIN_JPEG_1X1 = bytes(
    [
        0xFF,
        0xD8,
        0xFF,
        0xC0,
        0x00,
        0x11,
        0x08,
        0x00,
        0x01,
        0x00,
        0x01,
        0x03,
        0x01,
        0x11,
        0x00,
        0x02,
        0x11,
        0x01,
        0x03,
        0x11,
        0x01,
        0xFF,
        0xD9,
    ]
)


class TestBackendParityMatrix(unittest.TestCase):
    def test_backend_parity_matrix_generates_json_and_html(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "backend_parity_matrix.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            image = root / "000001.jpg"
            image.write_bytes(_MIN_JPEG_1X1)

            payload = [
                {
                    "image": str(image.resolve()),
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        }
                    ],
                }
            ]

            backends = ("torch", "onnxrt", "trt", "opencv_dnn", "custom_cpp")
            args = []
            for backend in backends:
                p = root / f"pred_{backend}.json"
                p.write_text(json.dumps(payload), encoding="utf-8")
                args.extend(["--backend-predictions", f"{backend}={p.relative_to(repo_root)}"])

            run_dir = root / "run"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    *args,
                    "--reference-backend",
                    "torch",
                    "--image-size",
                    "1,1",
                    "--run-dir",
                    str(run_dir.relative_to(repo_root)),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"backend_parity_matrix.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            report_json = run_dir / "reports" / "backend_parity_matrix.json"
            report_html = run_dir / "reports" / "backend_parity_matrix.html"
            self.assertTrue(report_json.is_file())
            self.assertTrue(report_html.is_file())

            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertTrue(report.get("ok"))
            self.assertEqual(report.get("reference_backend"), "torch")
            self.assertEqual(len(report.get("matrix") or []), 4)
            self.assertIsInstance(report.get("fixed_input_fingerprint"), str)
            self.assertEqual(len(report.get("fixed_input_fingerprint")), 64)

    def test_backend_parity_matrix_requires_all_backends(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "backend_parity_matrix.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            image = root / "000001.jpg"
            image.write_bytes(_MIN_JPEG_1X1)
            payload = [
                {
                    "image": str(image.resolve()),
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        }
                    ],
                }
            ]

            backends = ("torch", "onnxrt", "trt", "opencv_dnn")
            args = []
            for backend in backends:
                p = root / f"pred_{backend}.json"
                p.write_text(json.dumps(payload), encoding="utf-8")
                args.extend(["--backend-predictions", f"{backend}={p.relative_to(repo_root)}"])

            proc = subprocess.run(
                [sys.executable, str(script), *args],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("missing backend predictions", proc.stderr)


if __name__ == "__main__":
    unittest.main()
