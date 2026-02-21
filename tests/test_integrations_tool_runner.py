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


if __name__ == "__main__":
    unittest.main()
