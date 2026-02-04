import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_trt_engine


class _DummyInput:
    def __init__(self, name):
        self.name = name


class _DummyNetwork:
    def __init__(self, names):
        self._inputs = [_DummyInput(n) for n in names]

    @property
    def num_inputs(self):
        return len(self._inputs)

    def get_input(self, idx):
        return self._inputs[idx]


class TestBuildTrtEngine(unittest.TestCase):
    def test_resolve_input_name_prefers_requested(self):
        net = _DummyNetwork(["images", "other"])
        name = build_trt_engine._resolve_input_name(net, "images")
        self.assertEqual(name, "images")

    def test_resolve_input_name_falls_back(self):
        net = _DummyNetwork(["data"])
        name = build_trt_engine._resolve_input_name(net, "images")
        self.assertEqual(name, "data")

    def test_resolve_input_name_no_inputs(self):
        net = _DummyNetwork([])
        name = build_trt_engine._resolve_input_name(net, "images")
        self.assertEqual(name, "images")


if __name__ == "__main__":
    unittest.main()
