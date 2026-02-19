import unittest
from pathlib import Path


class TestAdapterStarters(unittest.TestCase):
    def test_required_adapter_starter_files_exist(self):
        repo_root = Path(__file__).resolve().parents[1]
        starters = {
            "mmdet": repo_root / "examples" / "adapter_starters" / "mmdet_adapter_starter.py",
            "detectron2": repo_root / "examples" / "adapter_starters" / "detectron2_adapter_starter.py",
            "ultralytics": repo_root / "examples" / "adapter_starters" / "ultralytics_adapter_starter.py",
            "rtdetr": repo_root / "examples" / "adapter_starters" / "rtdetr_adapter_starter.py",
            "opencv_dnn": repo_root / "examples" / "adapter_starters" / "opencv_dnn_adapter_starter.py",
            "custom_cpp": repo_root / "examples" / "adapter_starters" / "custom_cpp_route_starter.md",
        }

        for key, path in starters.items():
            self.assertTrue(path.is_file(), f"missing starter for {key}: {path}")


if __name__ == "__main__":
    unittest.main()
