import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.dataset import build_manifest


class TestDatasetUltralyticsCompat(unittest.TestCase):
    def _write_split(self, root: Path, split: str) -> None:
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / "000000.jpg").write_bytes(b"")
        (labels / "000000.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def test_autopick_train_split(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            dataset_root = Path(td) / "dataset"
            self._write_split(dataset_root, "train")

            manifest = build_manifest(dataset_root)
            self.assertEqual(manifest.get("split"), "train")
            self.assertEqual(len(manifest.get("images") or []), 1)

    def test_autopick_val_split(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            dataset_root = Path(td) / "dataset"
            self._write_split(dataset_root, "train")
            self._write_split(dataset_root, "val")

            manifest = build_manifest(dataset_root)
            self.assertEqual(manifest.get("split"), "val")
            self.assertEqual(len(manifest.get("images") or []), 1)

    def test_data_yaml_descriptor(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            self._write_split(dataset_root, "train")
            self._write_split(dataset_root, "val")

            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "\n".join(
                    [
                        f"path: {dataset_root}",
                        "train: images/train",
                        "val: images/val",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            manifest = build_manifest(data_yaml)
            self.assertEqual(manifest.get("split"), "val")
            self.assertEqual(len(manifest.get("images") or []), 1)

            manifest_train = build_manifest(data_yaml, split="train")
            self.assertEqual(manifest_train.get("split"), "train")
            self.assertEqual(len(manifest_train.get("images") or []), 1)


if __name__ == "__main__":
    unittest.main()

