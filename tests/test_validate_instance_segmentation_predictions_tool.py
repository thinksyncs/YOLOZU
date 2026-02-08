import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestValidateInstanceSegmentationPredictionsTool(unittest.TestCase):
    def test_validate_instance_segmentation_predictions_accepts_wrapper_and_mapping(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "validate_instance_segmentation_predictions.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)

            wrapper_path = root / "preds_wrapper.json"
            wrapper_path.write_text(
                json.dumps(
                    {
                        "predictions": [{"image": "a.jpg", "instances": [{"class_id": 0, "score": 0.9, "mask": "a.png"}]}],
                        "meta": {"note": "ok"},
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                ["python3", str(script), str(wrapper_path.relative_to(repo_root))],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"validate_instance_segmentation_predictions.py failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertIn("OK:", proc.stdout + proc.stderr)

            mapping_path = root / "preds_mapping.json"
            mapping_path.write_text(
                json.dumps(
                    {
                        "a.jpg": {"instances": [{"class_id": 0, "score": 0.9, "mask": "a.png"}]},
                        "b.jpg": [{"class_id": 1, "score": 0.8, "mask": "b.png"}],
                    }
                ),
                encoding="utf-8",
            )
            proc2 = subprocess.run(
                ["python3", str(script), str(mapping_path.relative_to(repo_root))],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc2.returncode != 0:
                self.fail(f"validate_instance_segmentation_predictions.py failed:\n{proc2.stdout}\n{proc2.stderr}")
            self.assertIn("OK:", proc2.stdout + proc2.stderr)


if __name__ == "__main__":
    unittest.main()

