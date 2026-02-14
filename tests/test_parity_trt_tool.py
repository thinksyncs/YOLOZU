import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import check_predictions_parity_trt


class TestParityTrtTool(unittest.TestCase):
    def test_append_arg_skips_none(self):
        args = ["--a", "1"]
        check_predictions_parity_trt._append_arg(args, "--b", None)
        self.assertEqual(args, ["--a", "1"])

    def test_append_arg_adds_value(self):
        args = []
        check_predictions_parity_trt._append_arg(args, "--b", 2)
        self.assertEqual(args, ["--b", "2"])


if __name__ == "__main__":
    unittest.main()
