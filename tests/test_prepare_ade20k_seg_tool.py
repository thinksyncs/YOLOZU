import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestPrepareADE20KSegTool(unittest.TestCase):
    def test_prepare_ade20k_seg_copy_writes_dataset_json(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "prepare_ade20k_seg.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            ade_root = root / "ade"
            challenge_root = ade_root / "ADEChallengeData2016"
            img_dir = challenge_root / "images" / "training"
            ann_dir = challenge_root / "annotations" / "training"
            img_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)

            sample_id = "ADE_train_00000001"
            (img_dir / f"{sample_id}.jpg").write_bytes(b"")
            (ann_dir / f"{sample_id}.png").write_bytes(b"")

            out_root = root / "out"
            proc = subprocess.run(
                [
                    "python3",
                    str(script),
                    "--ade20k-root",
                    str(ade_root),
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
                self.fail(f"prepare_ade20k_seg.py failed:\n{proc.stdout}\n{proc.stderr}")

            dataset_json = out_root / "dataset.json"
            self.assertTrue(dataset_json.is_file())
            payload = json.loads(dataset_json.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("dataset"), "ade20k")
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

