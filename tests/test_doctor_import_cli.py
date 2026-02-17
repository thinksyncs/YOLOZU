import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestDoctorImportCLI(unittest.TestCase):
    def _run(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "yolozu", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )

    def test_doctor_import_dataset_coco_instances_stdout(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            images_dir = root / "images" / "val2017"
            ann_dir = root / "annotations"
            images_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)

            instances = {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 64, "height": 32}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 20], "iscrowd": 0}],
                "categories": [{"id": 7, "name": "thing"}],
            }
            instances_path = ann_dir / "instances_val2017.json"
            instances_path.write_text(json.dumps(instances), encoding="utf-8")

            proc = self._run(
                [
                    "doctor",
                    "import",
                    "--dataset-from",
                    "coco-instances",
                    "--instances",
                    str(instances_path),
                    "--images-dir",
                    str(images_dir),
                    "--split",
                    "val2017",
                    "--output",
                    "-",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"doctor import (dataset) failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("kind"), "yolozu_doctor_import")
            ds = payload.get("dataset") or {}
            self.assertEqual(ds.get("from"), "coco-instances")
            counts = ds.get("counts") or {}
            self.assertEqual(int(counts.get("images")), 1)

    def test_doctor_import_config_ultralytics_stdout(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            args_yaml = root / "args.yaml"
            args_yaml.write_text("imgsz: 640\nbatch: 16\nepochs: 1\nlr0: 0.01\n", encoding="utf-8")

            proc = self._run(
                [
                    "doctor",
                    "import",
                    "--config-from",
                    "ultralytics",
                    "--args",
                    str(args_yaml),
                    "--output",
                    "-",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"doctor import (config) failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(proc.stdout)
            cfg = payload.get("config") or {}
            train = cfg.get("train_config") or {}
            self.assertEqual(train.get("format"), "yolozu_train_config_v1")
            self.assertEqual(int(train.get("batch")), 16)


if __name__ == "__main__":
    unittest.main()

