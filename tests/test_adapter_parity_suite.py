import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestAdapterParitySuite(unittest.TestCase):
    def test_adapter_parity_suite_supports_all_required_adapters(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "adapter_parity_suite.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            image_path = (root / "000001.jpg").resolve()
            image_path.write_bytes(
                bytes(
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
            )

            pred_payload = [
                {
                    "image": str(image_path),
                    "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
                }
            ]

            adapter_keys = ["mmdet", "detectron2", "ultralytics", "rtdetr", "opencv_dnn", "custom_cpp"]
            args = []
            for key in adapter_keys:
                p = root / f"pred_{key}.json"
                p.write_text(json.dumps(pred_payload), encoding="utf-8")
                args.extend(["--adapter-predictions", f"{key}={p.relative_to(repo_root)}"])

            out_path = root / "adapter_parity_suite.json"
            cmd = [
                sys.executable,
                str(script),
                *args,
                "--reference-adapter",
                "rtdetr",
                "--image-size",
                "1,1",
                "--output",
                str(out_path.relative_to(repo_root)),
            ]
            proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if proc.returncode != 0:
                self.fail(f"adapter_parity_suite failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(report.get("ok"))
            self.assertEqual(report.get("reference_adapter"), "rtdetr")
            self.assertEqual(sorted(report.get("supported_adapters") or []), sorted(adapter_keys))
            self.assertEqual(len(report.get("comparisons") or []), 5)


if __name__ == "__main__":
    unittest.main()
