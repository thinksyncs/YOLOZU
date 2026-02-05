import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestRunBaselineReport(unittest.TestCase):
    def test_unified_report_schema_dummy(self):
        from PIL import Image

        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images" / "train2017").mkdir(parents=True)
            (root / "labels" / "train2017").mkdir(parents=True)

            img = root / "images" / "train2017" / "0001.jpg"
            Image.new("RGB", (10, 10)).save(img)
            (root / "labels" / "train2017" / "0001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            out_path = root / "baseline.json"
            cmd = [
                sys.executable,
                str(repo_root / "tools" / "run_baseline.py"),
                "--adapter",
                "dummy",
                "--dataset",
                str(root),
                "--split",
                "train2017",
                "--output",
                str(out_path),
                "--no-scenarios",
            ]
            subprocess.check_call(cmd)

            data = json.loads(out_path.read_text())
            self.assertEqual(data.get("schema_version"), 1)
            self.assertIn("meta", data)
            self.assertIn("speed", data)
            self.assertIn("summary", data)
            self.assertIn("predictions", data)
            self.assertIn("coco", data)
            self.assertIsInstance(data["predictions"].get("hash_sha256"), str)


if __name__ == "__main__":
    unittest.main()
