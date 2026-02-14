import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestMakeSubsetDataset(unittest.TestCase):
    def _make_dummy_dataset(self, root: Path) -> Path:
        dataset_root = root / "dataset"
        images = dataset_root / "images" / "val2017"
        labels = dataset_root / "labels" / "val2017"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            (images / f"{i:06d}.jpg").write_bytes(b"")
            if i % 2 == 0:
                (labels / f"{i:06d}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        return dataset_root

    def test_make_subset_dataset_is_deterministic(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "make_subset_dataset.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._make_dummy_dataset(root)

            out1 = root / "out1"
            out2 = root / "out2"
            cmd = [
                sys.executable,
                str(script),
                "--dataset",
                str(dataset_root),
                "--split",
                "val2017",
                "--n",
                "5",
                "--seed",
                "123",
                "--strategy",
                "hash",
            ]

            proc1 = subprocess.run(cmd + ["--out", str(out1)], cwd=str(repo_root), capture_output=True, text=True)
            self.assertEqual(proc1.returncode, 0, proc1.stdout + proc1.stderr)

            proc2 = subprocess.run(cmd + ["--out", str(out2)], cwd=str(repo_root), capture_output=True, text=True)
            self.assertEqual(proc2.returncode, 0, proc2.stdout + proc2.stderr)

            list1 = (out1 / "subset_images.txt").read_text(encoding="utf-8")
            list2 = (out2 / "subset_images.txt").read_text(encoding="utf-8")
            self.assertEqual(list1, list2)

            payload = json.loads((out1 / "subset.json").read_text(encoding="utf-8"))
            images = payload.get("images") or []
            self.assertEqual(len(images), 5)
            sha = payload.get("images_sha256")
            self.assertIsInstance(sha, str)
            self.assertTrue(sha)

            out_images = list((out1 / "images" / "val2017").glob("*.jpg"))
            self.assertEqual(len(out_images), 5)


if __name__ == "__main__":
    unittest.main()

