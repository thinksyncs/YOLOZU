import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestPrepareVocSegTool(unittest.TestCase):
    def test_prepare_voc_seg_copy_writes_dataset_json(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "prepare_voc_seg.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            voc_root = root / "VOCdevkit" / "VOC2012"
            (voc_root / "JPEGImages").mkdir(parents=True, exist_ok=True)
            (voc_root / "SegmentationClass").mkdir(parents=True, exist_ok=True)
            split_dir = voc_root / "ImageSets" / "Segmentation"
            split_dir.mkdir(parents=True, exist_ok=True)

            sample_id = "2007_000001"
            (split_dir / "train.txt").write_text(f"{sample_id}\n", encoding="utf-8")

            (voc_root / "JPEGImages" / f"{sample_id}.jpg").write_bytes(b"")
            (voc_root / "SegmentationClass" / f"{sample_id}.png").write_bytes(b"")

            out_root = root / "out"
            proc = subprocess.run(
                [
                    "python3",
                    str(script),
                    "--voc-root",
                    str(voc_root.parent),
                    "--year",
                    "2012",
                    "--split",
                    "train",
                    "--out",
                    str(out_root),
                    "--mode",
                    "copy",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"prepare_voc_seg.py failed:\n{proc.stdout}\n{proc.stderr}")

            dataset_json = out_root / "dataset.json"
            self.assertTrue(dataset_json.is_file())
            payload = json.loads(dataset_json.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("dataset"), "pascal_voc")
            self.assertEqual(payload.get("task"), "semantic_segmentation")
            self.assertEqual(payload.get("split"), "train")
            self.assertEqual(payload.get("mode"), "copy")
            self.assertEqual(payload.get("path_type"), "relative")

            samples = payload.get("samples") or []
            self.assertEqual(len(samples), 1)
            self.assertTrue(str(samples[0]["image"]).startswith("images/train/"))
            self.assertTrue(str(samples[0]["mask"]).startswith("masks/train/"))

            copied_image = out_root / samples[0]["image"]
            copied_mask = out_root / samples[0]["mask"]
            self.assertTrue(copied_image.is_file())
            self.assertTrue(copied_mask.is_file())


if __name__ == "__main__":
    unittest.main()

