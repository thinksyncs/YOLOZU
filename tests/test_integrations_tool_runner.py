import unittest
from unittest.mock import patch

from yolozu.integrations import tool_runner


class TestIntegrationToolRunner(unittest.TestCase):
    def test_a1_validate_predictions_strict_default(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "validate_predictions", "summary": "ok"}
            out = tool_runner.validate_predictions("data/smoke/predictions/predictions_dummy.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "validate_predictions",
            ["validate", "predictions", "data/smoke/predictions/predictions_dummy.json", "--strict"],
        )

    def test_a2_validate_dataset_builds_expected_args(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "validate_dataset", "summary": "ok"}
            out = tool_runner.validate_dataset("data/smoke", split="val", strict=True, mode="warn")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "validate_dataset",
            ["validate", "dataset", "data/smoke", "--mode", "warn", "--split", "val", "--strict"],
        )

    def test_a3_convert_dataset_manifest_mode_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "convert_dataset", "summary": "ok"}
            out = tool_runner.convert_dataset(
                from_format="ultralytics",
                output="reports/converted_dataset",
                data="data/smoke",
            )

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "convert_dataset",
            [
                "migrate",
                "dataset",
                "--from",
                "ultralytics",
                "--output",
                "reports/converted_dataset",
                "--mode",
                "manifest",
                "--data",
                "data/smoke",
                "--force",
            ],
        )

    def test_b4_predict_images_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "predict_images", "summary": "ok"}
            out = tool_runner.predict_images("data/smoke/images/val")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "predict_images",
            [
                "predict-images",
                "--backend",
                "dummy",
                "--input-dir",
                "data/smoke/images/val",
                "--output",
                "reports/mcp_predict_images.json",
                "--dry-run",
                "--strict",
                "--force",
            ],
            artifacts={"predictions": "reports/mcp_predict_images.json"},
        )

    def test_b5_parity_check_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "parity_check", "summary": "ok"}
            out = tool_runner.parity_check("reports/ref.json", "reports/cand.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "parity_check",
            [
                "parity",
                "--reference",
                "reports/ref.json",
                "--candidate",
                "reports/cand.json",
                "--iou-thresh",
                "0.5",
                "--score-atol",
                "1e-06",
                "--bbox-atol",
                "0.0001",
            ],
        )

    def test_b6_calibrate_predictions_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "calibrate_predictions", "summary": "ok"}
            out = tool_runner.calibrate_predictions("data/smoke", "reports/predictions.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "calibrate_predictions",
            [
                "calibrate",
                "--method",
                "fracal",
                "--dataset",
                "data/smoke",
                "--task",
                "auto",
                "--predictions",
                "reports/predictions.json",
                "--output",
                "reports/mcp_calibrated_predictions.json",
                "--output-report",
                "reports/mcp_calibration_report.json",
                "--force",
            ],
            artifacts={
                "predictions": "reports/mcp_calibrated_predictions.json",
                "report": "reports/mcp_calibration_report.json",
            },
        )

    def test_c7_eval_coco_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "eval_coco", "summary": "ok"}
            out = tool_runner.eval_coco("data/smoke", "data/smoke/predictions/predictions_dummy.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "eval_coco",
            [
                "eval-coco",
                "--dataset",
                "data/smoke",
                "--predictions",
                "data/smoke/predictions/predictions_dummy.json",
                "--output",
                "reports/mcp_coco_eval.json",
                "--dry-run",
            ],
            artifacts={"report": "reports/mcp_coco_eval.json"},
        )

    def test_c8_eval_instance_seg_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "eval_instance_seg", "summary": "ok"}
            out = tool_runner.eval_instance_seg("data/smoke", "reports/instance_seg_predictions.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "eval_instance_seg",
            [
                "eval-instance-seg",
                "--dataset",
                "data/smoke",
                "--predictions",
                "reports/instance_seg_predictions.json",
                "--output",
                "reports/mcp_instance_seg_eval.json",
            ],
            artifacts={"report": "reports/mcp_instance_seg_eval.json"},
        )

    def test_c9_eval_long_tail_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "eval_long_tail", "summary": "ok"}
            out = tool_runner.eval_long_tail("data/smoke", "data/smoke/predictions/predictions_dummy.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "eval_long_tail",
            [
                "eval-long-tail",
                "--dataset",
                "data/smoke",
                "--predictions",
                "data/smoke/predictions/predictions_dummy.json",
                "--output",
                "reports/mcp_long_tail_eval.json",
            ],
            artifacts={"report": "reports/mcp_long_tail_eval.json"},
        )


if __name__ == "__main__":
    unittest.main()
