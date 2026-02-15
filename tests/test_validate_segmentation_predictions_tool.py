import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestValidateSegmentationPredictionsTool(unittest.TestCase):
    def test_validate_segmentation_predictions_accepts_wrapper_and_mapping(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "validate_segmentation_predictions.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)

            wrapper_path = root / "preds_wrapper.json"
            wrapper_path.write_text(
                json.dumps(
                    {
                        "predictions": [{"id": "a", "mask": "a.png"}],
                        "meta": {"note": "ok"},
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [sys.executable, str(script), str(wrapper_path.relative_to(repo_root))],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"validate_segmentation_predictions.py failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertIn("OK:", proc.stdout + proc.stderr)

            mapping_path = root / "preds_mapping.json"
            mapping_path.write_text(json.dumps({"a": "a.png", "b": "b.png"}), encoding="utf-8")
            proc2 = subprocess.run(
                [sys.executable, str(script), str(mapping_path.relative_to(repo_root))],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc2.returncode != 0:
                self.fail(f"validate_segmentation_predictions.py failed:\n{proc2.stdout}\n{proc2.stderr}")
            self.assertIn("OK:", proc2.stdout + proc2.stderr)


if __name__ == "__main__":
    unittest.main()
