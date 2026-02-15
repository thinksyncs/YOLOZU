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


class TestTuneGateWeightsTool(unittest.TestCase):
    def test_config_relative_paths_and_cli_override(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "tune_gate_weights.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            (dataset_root / "images" / "train2017").mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / "train2017").mkdir(parents=True, exist_ok=True)

            image_path = dataset_root / "images" / "train2017" / "000001.jpg"
            image_path.write_bytes(_MIN_JPEG_1X1)
            (dataset_root / "labels" / "train2017" / "000001.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            cfg_dir = root / "cfg"
            cfg_dir.mkdir(parents=True, exist_ok=True)

            preds_path = cfg_dir / "predictions.json"
            preds_path.write_text(
                json.dumps(
                    [
                        {
                            "image": str(image_path),
                            "detections": [
                                {
                                    "class_id": 0,
                                    "score": 0.9,
                                    "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            cfg_path = cfg_dir / "tune.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset": "../dataset",
                        "split": "train2017",
                        "predictions": "predictions.json",
                        "grid": {"det": [1.0], "tmp": [0.0], "unc": [0.0]},
                        "output_report": "reports/from_config.json",
                    }
                ),
                encoding="utf-8",
            )

            output_override = root / "reports" / "from_cli.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(cfg_path),
                    "--output-report",
                    str(output_override),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"tune_gate_weights.py failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(output_override.is_file())
            config_default_report = cfg_dir / "reports" / "from_config.json"
            self.assertFalse(config_default_report.exists())


if __name__ == "__main__":
    unittest.main()
