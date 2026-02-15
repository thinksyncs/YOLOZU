import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestCheckKeypointsParityTool(unittest.TestCase):
    def test_parity_ok_and_fail(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_keypoints_parity.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)

            image = str(root / "img.jpg")
            ref = root / "ref.json"
            cand = root / "cand.json"

            ref.write_text(
                json.dumps(
                    [
                        {
                            "image": image,
                            "detections": [
                                {
                                    "class_id": 0,
                                    "score": 0.9,
                                    "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4},
                                    "keypoints": [0.50, 0.50, 2, 0.60, 0.50, 2],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            cand.write_text(
                json.dumps(
                    [
                        {
                            "image": image,
                            "detections": [
                                {
                                    "class_id": 0,
                                    "score": 0.90000001,
                                    "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4},
                                    "keypoints": [0.5004, 0.50, 2, 0.60, 0.5002, 2],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            ok = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--reference",
                    str(ref),
                    "--candidate",
                    str(cand),
                    "--iou-thresh",
                    "0.99",
                    "--kp-atol",
                    "1e-3",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if ok.returncode != 0:
                self.fail(f"check_keypoints_parity.py expected ok:\n{ok.stdout}\n{ok.stderr}")

            bad = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--reference",
                    str(ref),
                    "--candidate",
                    str(cand),
                    "--iou-thresh",
                    "0.99",
                    "--kp-atol",
                    "1e-5",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(bad.returncode, 0, msg=f"expected non-zero:\n{bad.stdout}\n{bad.stderr}")


if __name__ == "__main__":
    unittest.main()
