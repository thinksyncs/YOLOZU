import tempfile
import unittest
from pathlib import Path

from yolozu.cli_args import (
    parse_image_size_arg,
    require_non_negative_float,
    require_non_negative_int,
    require_positive_int,
    resolve_input_path,
    resolve_output_path,
)


class TestCliArgs(unittest.TestCase):
    def test_parse_image_size_variants(self):
        self.assertEqual(parse_image_size_arg("640"), (640, 640))
        self.assertEqual(parse_image_size_arg("640,480"), (640, 480))
        self.assertEqual(parse_image_size_arg("640x480"), (640, 480))
        with self.assertRaises(ValueError):
            parse_image_size_arg("0")

    def test_numeric_validators(self):
        self.assertEqual(require_non_negative_int(0, flag_name="--x"), 0)
        self.assertEqual(require_positive_int(1, flag_name="--x"), 1)
        self.assertEqual(require_non_negative_float(0.0, flag_name="--x"), 0.0)
        with self.assertRaises(ValueError):
            require_non_negative_int(-1, flag_name="--x")
        with self.assertRaises(ValueError):
            require_positive_int(0, flag_name="--x")
        with self.assertRaises(ValueError):
            require_non_negative_float(-0.1, flag_name="--x")

    def test_resolve_input_path_prefers_config_dir_when_exists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cwd = root / "cwd"
            cfg_dir = root / "cfg"
            repo = root / "repo"
            cwd.mkdir()
            cfg_dir.mkdir()
            repo.mkdir()
            target = cfg_dir / "predictions.json"
            target.write_text("[]", encoding="utf-8")

            resolved = resolve_input_path(
                "predictions.json",
                cwd=cwd,
                repo_root=repo,
                config_dir=cfg_dir,
            )
            self.assertEqual(resolved, target)

    def test_resolve_input_path_falls_back_to_repo_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cwd = root / "cwd"
            repo = root / "repo"
            cwd.mkdir()
            repo.mkdir()
            target = repo / "dataset"
            target.mkdir()

            resolved = resolve_input_path("dataset", cwd=cwd, repo_root=repo)
            self.assertEqual(resolved, target)

    def test_resolve_output_path_uses_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            out = resolve_output_path("reports/out.json", cwd=cwd)
            self.assertEqual(out, cwd / "reports" / "out.json")


if __name__ == "__main__":
    unittest.main()
