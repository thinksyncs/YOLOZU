import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPrepareCityscapesSegTool(unittest.TestCase):
    def test_prepare_cityscapes_seg_copy_writes_dataset_json(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "prepare_cityscapes_seg.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            city_root = root / "cityscapes"
            img_dir = city_root / "leftImg8bit_trainvaltest" / "leftImg8bit" / "train" / "aachen"
            lab_dir = city_root / "gtFine_trainvaltest" / "gtFine" / "train" / "aachen"
            img_dir.mkdir(parents=True, exist_ok=True)
            lab_dir.mkdir(parents=True, exist_ok=True)

            img_name = "aachen_000000_000000_leftImg8bit.png"
            mask_name = "aachen_000000_000000_gtFine_labelTrainIds.png"
            (img_dir / img_name).write_bytes(b"")
            (lab_dir / mask_name).write_bytes(b"")

            out_root = root / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--cityscapes-root",
                    str(city_root),
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
                self.fail(f"prepare_cityscapes_seg.py failed:\n{proc.stdout}\n{proc.stderr}")

            dataset_json = out_root / "dataset.json"
            self.assertTrue(dataset_json.is_file())
            payload = json.loads(dataset_json.read_text())
            self.assertEqual(payload.get("dataset"), "cityscapes")
            self.assertEqual(payload.get("task"), "semantic_segmentation")
            self.assertEqual(payload.get("split"), "train")
            self.assertEqual(payload.get("label_type"), "labelTrainIds")
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
