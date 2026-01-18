import json
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.config import get_symmetry_spec, load_constraints, load_symmetry_map


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_load_constraints_defaults(self):
        cfg = load_constraints(self.repo_root / "constraints.yaml")
        self.assertIn("enabled", cfg)
        self.assertFalse(cfg["enabled"]["depth_prior"])
        self.assertIsInstance(cfg["table_plane"]["n"], list)

    def test_invalid_constraints(self):
        bad_yaml = "enabled:\n  depth_prior: nope\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yaml"
            path.write_text(bad_yaml)
            with self.assertRaises(ValueError):
                load_constraints(path)

    def test_symmetry_map_validation(self):
        bad_sym = {"cup": {"type": "Cn", "n": -1}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sym.json"
            path.write_text(json.dumps(bad_sym))
            with self.assertRaises(ValueError):
                load_symmetry_map(path)

    def test_symmetry_defaults(self):
        sym = {"1": {"type": "C2", "axis": [0.0, 0.0, 1.0]}, "cup": {"type": "none"}}
        self.assertIsNone(get_symmetry_spec(sym, "missing"))
        self.assertEqual(get_symmetry_spec(sym, 1), sym["1"])
        self.assertEqual(get_symmetry_spec(sym, "1"), sym["1"])
        self.assertEqual(get_symmetry_spec(sym, "cup"), sym["cup"])


if __name__ == "__main__":
    unittest.main()
