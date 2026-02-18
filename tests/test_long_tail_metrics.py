import unittest

from yolozu.long_tail_metrics import (
    build_fracal_stats,
    evaluate_long_tail_detection,
    fracal_calibrate_instance_segmentation,
    fracal_calibrate_predictions,
)

try:
    import numpy as _np  # noqa: F401
    from PIL import Image as _PILImage  # noqa: F401

    _HAS_SEG_DEPS = True
except Exception:
    _HAS_SEG_DEPS = False


def _mk_label(class_id: int, cx: float) -> dict:
    return {"class_id": int(class_id), "cx": float(cx), "cy": 0.5, "w": 0.2, "h": 0.2}


def _mk_det(class_id: int, score: float, cx: float) -> dict:
    return {
        "class_id": int(class_id),
        "score": float(score),
        "bbox": {"cx": float(cx), "cy": 0.5, "w": 0.2, "h": 0.2},
    }


class TestLongTailMetrics(unittest.TestCase):
    def _fixture(self):
        records = [
            {"image": "img1.jpg", "labels": [_mk_label(0, 0.30), _mk_label(1, 0.70)]},
            {"image": "img2.jpg", "labels": [_mk_label(0, 0.30)]},
            {"image": "img3.jpg", "labels": [_mk_label(0, 0.30), _mk_label(2, 0.70)]},
        ]
        preds = [
            {"image": "img1.jpg", "detections": [_mk_det(0, 0.90, 0.30), _mk_det(1, 0.40, 0.70)]},
            {"image": "img2.jpg", "detections": [_mk_det(0, 0.85, 0.30)]},
            {"image": "img3.jpg", "detections": [_mk_det(0, 0.88, 0.30), _mk_det(2, 0.40, 0.70)]},
        ]
        return records, preds

    def test_fracal_boosts_tail_vs_head(self):
        records, preds = self._fixture()
        calibrated, report = fracal_calibrate_predictions(records, preds, alpha=0.8, strength=1.0)

        head_before = preds[0]["detections"][0]["score"]
        head_after = calibrated[0]["detections"][0]["score"]
        tail_before = preds[0]["detections"][1]["score"]
        tail_after = calibrated[0]["detections"][1]["score"]

        self.assertLess(head_after, head_before)
        self.assertGreater(tail_after, tail_before)
        self.assertEqual(report.get("method"), "fracal")

    def test_long_tail_report_contains_required_sections(self):
        records, preds = self._fixture()
        report = evaluate_long_tail_detection(records, preds)

        self.assertIn("per_class", report)
        self.assertIn("frequency_bins", report)
        self.assertIn("metrics", report)
        self.assertIn("macro", report["metrics"])
        self.assertIn("calibration", report["metrics"])

        self.assertTrue(report["per_class"])
        self.assertIn("head", report["frequency_bins"])
        self.assertIn("medium", report["frequency_bins"])
        self.assertIn("tail", report["frequency_bins"])

        macro = report["metrics"]["macro"]
        self.assertIn("ap50", macro)
        self.assertIn("ar50", macro)

        calib = report["metrics"]["calibration"]
        self.assertIn("ece", calib)
        self.assertIn("confidence_bias", calib)

    def test_fracal_uses_precomputed_class_counts_for_bbox(self):
        records, preds = self._fixture()
        counts = {"0": 100, "1": 1, "2": 1}
        calibrated, report = fracal_calibrate_predictions(records, preds, alpha=1.0, strength=1.0, class_counts=counts)

        head_before = preds[0]["detections"][0]["score"]
        head_after = calibrated[0]["detections"][0]["score"]
        tail_before = preds[0]["detections"][1]["score"]
        tail_after = calibrated[0]["detections"][1]["score"]

        self.assertLess(head_after, head_before)
        self.assertGreater(tail_after, tail_before)
        self.assertEqual(report["class_counts"], {"0": 100, "1": 1, "2": 1})

    @unittest.skipUnless(_HAS_SEG_DEPS, "instance-seg calibration requires numpy and Pillow")
    def test_build_fracal_stats_for_seg_and_calibrate_instances(self):
        records = [
            {
                "image": "img1.png",
                "mask": [
                    [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                ],
                "mask_classes": [0],
            },
            {
                "image": "img2.png",
                "mask": [
                    [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                ],
                "mask_classes": [1],
            },
        ]
        preds = [
            {
                "image": "img1.png",
                "instances": [
                    {"class_id": 0, "score": 0.9, "mask": [[1, 1, 0], [1, 0, 0], [0, 0, 0]]},
                    {"class_id": 1, "score": 0.4, "mask": [[1, 1, 1], [0, 0, 0], [0, 0, 0]]},
                ],
            }
        ]

        stats = build_fracal_stats(records, task="seg")
        self.assertEqual(stats.get("task"), "seg")
        self.assertEqual(stats.get("class_counts"), {"0": 1, "1": 1})

        calibrated, report = fracal_calibrate_instance_segmentation(
            records,
            preds,
            alpha=0.6,
            strength=1.0,
            class_counts={"0": 10, "1": 1},
        )
        self.assertEqual(report.get("method"), "fracal")
        self.assertEqual(report.get("class_counts"), {"0": 10, "1": 1})
        self.assertIn("instances", calibrated[0])
        self.assertEqual(len(calibrated[0]["instances"]), 2)
        self.assertLess(calibrated[0]["instances"][0]["score"], preds[0]["instances"][0]["score"])
        self.assertGreater(calibrated[0]["instances"][1]["score"], preds[0]["instances"][1]["score"])


if __name__ == "__main__":
    unittest.main()
