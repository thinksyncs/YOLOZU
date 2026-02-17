import subprocess
import sys
import unittest
from pathlib import Path


class TestTrainMinimalWrapperImport(unittest.TestCase):
    def test_train_minimal_wrapper_importable(self):
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import rtdetr_pose.tools.train_minimal as tm; assert callable(getattr(tm, 'main', None))",
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"import failed:\n{proc.stdout}\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main()

