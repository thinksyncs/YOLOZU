import unittest


class TestCanonicalSchema(unittest.TestCase):
    def test_sample_record_to_dict(self):
        from yolozu.canonical import BBox, Label, SampleRecord

        rec = SampleRecord(
            image_path="a.jpg",
            width=640,
            height=480,
            labels=[Label(class_id=0, bbox=BBox(cx=0.5, cy=0.5, w=0.2, h=0.2))],
        )
        out = rec.to_record_dict()
        self.assertEqual(out["image"], "a.jpg")
        self.assertEqual(out["image_hw"], [480, 640])
        self.assertEqual(out["labels"][0]["class_id"], 0)
        self.assertAlmostEqual(float(out["labels"][0]["cx"]), 0.5, places=6)

    def test_train_config_to_dict_has_format(self):
        from yolozu.canonical import TrainConfig

        cfg = TrainConfig(imgsz=640, batch=8, epochs=1, lr=1e-3)
        out = cfg.to_dict()
        self.assertEqual(out.get("format"), "yolozu_train_config_v1")
        self.assertEqual(int(out.get("batch")), 8)


if __name__ == "__main__":
    unittest.main()

