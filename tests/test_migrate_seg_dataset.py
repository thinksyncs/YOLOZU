import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.migrate import migrate_seg_dataset_descriptor, migrate_ultralytics_dataset_wrapper
from yolozu.segmentation_dataset import load_seg_dataset_descriptor


class TestMigrateSegDataset(unittest.TestCase):
    def test_migrate_ultralytics_dataset_wrapper_with_task_segment(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            base = Path(td)
            dataset_root = base / "ultra"
            (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
            (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

            (dataset_root / "images" / "val" / "0001.jpg").write_bytes(b"")
            (dataset_root / "labels" / "val" / "0001.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n",
                encoding="utf-8",
            )

            data_yaml = base / "data.yaml"
            data_yaml.write_text(
                "\n".join(
                    [
                        f"path: {dataset_root}",
                        "train: images/train",
                        "val: images/val",
                        "task: segment",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            out_dir = base / "wrapper"
            out_path = migrate_ultralytics_dataset_wrapper(data_yaml=data_yaml, output=out_dir, force=False)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("label_format"), "segment")

            from yolozu.dataset import build_manifest

            manifest = build_manifest(out_dir)
            labels = (manifest.get("images") or [])[0]["labels"]
            self.assertIn("polygon", labels[0])

    def test_migrate_voc_descriptor(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "VOC2012"
            (root / "JPEGImages").mkdir(parents=True, exist_ok=True)
            (root / "SegmentationClass").mkdir(parents=True, exist_ok=True)
            (root / "ImageSets" / "Segmentation").mkdir(parents=True, exist_ok=True)

            (root / "JPEGImages" / "0001.jpg").write_bytes(b"")
            (root / "SegmentationClass" / "0001.png").write_bytes(b"")
            (root / "ImageSets" / "Segmentation" / "val.txt").write_text("0001\n", encoding="utf-8")

            out = Path(td) / "voc_seg.json"
            migrate_seg_dataset_descriptor(from_format="voc", root=root, split="val", output=out, force=False)
            desc = load_seg_dataset_descriptor(out)
            self.assertEqual(desc.task, "semantic_segmentation")
            self.assertEqual(desc.split, "val")
            self.assertEqual(len(desc.samples), 1)
            self.assertEqual(desc.samples[0].sample_id, "0001")
            self.assertIsNotNone(desc.samples[0].mask)

    def test_migrate_cityscapes_descriptor(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cityscapes"
            img_dir = root / "leftImg8bit" / "val" / "foo"
            mask_dir = root / "gtFine" / "val" / "foo"
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            base = "foo_000000_000001"
            img_path = img_dir / f"{base}_leftImg8bit.png"
            mask_path = mask_dir / f"{base}_gtFine_labelTrainIds.png"
            img_path.write_bytes(b"")
            mask_path.write_bytes(b"")

            out = Path(td) / "city_seg.json"
            migrate_seg_dataset_descriptor(from_format="cityscapes", root=root, split="val", output=out, force=False)
            desc = load_seg_dataset_descriptor(out)
            self.assertEqual(desc.dataset, "cityscapes")
            self.assertEqual(len(desc.samples), 1)
            self.assertEqual(desc.samples[0].sample_id, base)

    def test_migrate_ade20k_descriptor(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ade20k"
            (root / "images" / "validation").mkdir(parents=True, exist_ok=True)
            (root / "annotations" / "validation").mkdir(parents=True, exist_ok=True)
            (root / "images" / "validation" / "ADE_val_0001.jpg").write_bytes(b"")
            (root / "annotations" / "validation" / "ADE_val_0001.png").write_bytes(b"")

            out = Path(td) / "ade_seg.json"
            migrate_seg_dataset_descriptor(from_format="ade20k", root=root, split="val", output=out, force=False)
            desc = load_seg_dataset_descriptor(out)
            self.assertEqual(desc.dataset, "ade20k")
            self.assertEqual(len(desc.samples), 1)


if __name__ == "__main__":
    unittest.main()

