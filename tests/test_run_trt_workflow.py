import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_trt_workflow


class TestRunTrtWorkflow(unittest.TestCase):
    def test_build_commands_include_build_and_parity(self):
        args = run_trt_workflow._parse_args(
            [
                "--onnx",
                "model.onnx",
                "--engine",
                "model.plan",
                "--dataset",
                "data/coco",
                "--reference",
                "reports/pred_ref.json",
            ]
        )
        cmds = run_trt_workflow._build_commands(args)
        self.assertEqual(len(cmds), 2)
        self.assertIn("build_trt_engine.py", " ".join(cmds[0]))
        self.assertIn("check_predictions_parity_trt.py", " ".join(cmds[1]))

    def test_skip_build(self):
        args = run_trt_workflow._parse_args(
            [
                "--onnx",
                "model.onnx",
                "--engine",
                "model.plan",
                "--dataset",
                "data/coco",
                "--reference",
                "reports/pred_ref.json",
                "--skip-build",
            ]
        )
        cmds = run_trt_workflow._build_commands(args)
        self.assertEqual(len(cmds), 1)
        self.assertIn("check_predictions_parity_trt.py", " ".join(cmds[0]))

    def test_skip_parity(self):
        args = run_trt_workflow._parse_args(
            [
                "--onnx",
                "model.onnx",
                "--engine",
                "model.plan",
                "--dataset",
                "data/coco",
                "--reference",
                "reports/pred_ref.json",
                "--skip-parity",
            ]
        )
        cmds = run_trt_workflow._build_commands(args)
        self.assertEqual(len(cmds), 1)
        self.assertIn("build_trt_engine.py", " ".join(cmds[0]))


if __name__ == "__main__":
    unittest.main()
