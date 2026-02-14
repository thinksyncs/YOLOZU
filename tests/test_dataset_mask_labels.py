import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except Exception:  # pragma: no cover
    np = None
    Image = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.dataset import build_manifest


def _write_png(path: Path, arr: "np.ndarray", *, mode: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        raise RuntimeError("Pillow is required for this test")
    img = Image.fromarray(arr, mode=mode)
    img.save(path)


@unittest.skipUnless(np is not None and Image is not None, "requires numpy and Pillow")
class TestDatasetMaskLabels(unittest.TestCase):
    def _make_root(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "images" / "train2017").mkdir(parents=True)
        (root / "labels" / "train2017").mkdir(parents=True)
        (root / "masks" / "train2017").mkdir(parents=True)
        return root

    def _write_image(self, root: Path, stem: str, *, size=(10, 10)) -> Path:
        img_path = root / "images" / "train2017" / f"{stem}.png"
        img = Image.new("RGB", size, (255, 255, 255))
        img.save(img_path)
        return img_path

    def test_color_mask_derives_bbox_and_class(self):
        root = self._make_root()
        stem = "000001"
        self._write_image(root, stem)

        mask = np.zeros((10, 10, 3), dtype=np.uint8)
        mask[3:6, 2:5] = np.array([255, 0, 0], dtype=np.uint8)
        _write_png(root / "masks" / "train2017" / f"{stem}.png", mask)

        meta = {
            "mask_path": f"masks/train2017/{stem}.png",
            "mask_format": "color",
            "mask_class_map": {"255,0,0": 5},
        }
        (root / "labels" / "train2017" / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")

        manifest = build_manifest(root, split="train2017")
        rec = manifest["images"][0]
        labels = rec["labels"]
        self.assertEqual(len(labels), 1)
        lbl = labels[0]
        self.assertEqual(lbl["class_id"], 5)
        self.assertAlmostEqual(lbl["cx"], 0.35, places=6)
        self.assertAlmostEqual(lbl["cy"], 0.45, places=6)
        self.assertAlmostEqual(lbl["w"], 0.3, places=6)
        self.assertAlmostEqual(lbl["h"], 0.3, places=6)

    def test_instance_mask_derives_instances_with_fallback_class(self):
        root = self._make_root()
        stem = "000002"
        self._write_image(root, stem)

        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:3, 1:4] = 7
        mask[6:9, 6:8] = 9
        _write_png(root / "masks" / "train2017" / f"{stem}.png", mask, mode="L")

        meta = {
            "mask_path": f"masks/train2017/{stem}.png",
            "mask_format": "instance",
            "mask_class_id": 3,
        }
        (root / "labels" / "train2017" / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")

        manifest = build_manifest(root, split="train2017")
        rec = [r for r in manifest["images"] if Path(r["image"]).stem == stem][0]
        labels = sorted(rec["labels"], key=lambda d: (d["cx"], d["cy"]))
        self.assertEqual(len(labels), 2)
        self.assertEqual({l["class_id"] for l in labels}, {3})

        # Instance 7: x=[1..3], y=[1..2]
        self.assertAlmostEqual(labels[0]["cx"], 0.25, places=6)
        self.assertAlmostEqual(labels[0]["cy"], 0.20, places=6)
        self.assertAlmostEqual(labels[0]["w"], 0.30, places=6)
        self.assertAlmostEqual(labels[0]["h"], 0.20, places=6)

        # Instance 9: x=[6..7], y=[6..8]
        self.assertAlmostEqual(labels[1]["cx"], 0.70, places=6)
        self.assertAlmostEqual(labels[1]["cy"], 0.75, places=6)
        self.assertAlmostEqual(labels[1]["w"], 0.20, places=6)
        self.assertAlmostEqual(labels[1]["h"], 0.30, places=6)

    def test_mask_path_list_supports_mask_classes_and_instances(self):
        root = self._make_root()
        stem = "000003"
        self._write_image(root, stem)

        mask_a = np.zeros((10, 10), dtype=np.uint8)
        mask_a[1:3, 1:3] = 255
        mask_a[7:9, 7:9] = 255
        _write_png(root / "masks" / "train2017" / "a.png", mask_a, mode="L")

        mask_b = np.zeros((10, 10), dtype=np.uint8)
        mask_b[4:8, 2:4] = 255
        _write_png(root / "masks" / "train2017" / "b.png", mask_b, mode="L")

        meta = {
            "mask_path": ["masks/train2017/a.png", "masks/train2017/b.png"],
            "mask_classes": [2, 4],
            "mask_instances": True,
        }
        (root / "labels" / "train2017" / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")

        manifest = build_manifest(root, split="train2017")
        rec = [r for r in manifest["images"] if Path(r["image"]).stem == stem][0]
        labels = rec["labels"]
        self.assertEqual(len(labels), 3)
        class_ids = sorted([int(l["class_id"]) for l in labels])
        self.assertEqual(class_ids, [2, 2, 4])


if __name__ == "__main__":
    unittest.main()
