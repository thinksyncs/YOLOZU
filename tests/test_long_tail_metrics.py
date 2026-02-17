import unittest

from yolozu.long_tail_metrics import evaluate_long_tail_detection, fracal_calibrate_predictions


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


if __name__ == "__main__":
    unittest.main()
