import unittest
from pathlib import Path


class TestRunRecord(unittest.TestCase):
    def test_build_run_record_outside_git(self):
        from yolozu.run_record import build_run_record

        tmp_root = Path("/tmp") / "yolozu_run_record_test_no_git"
        # Intentionally do not create a .git dir.
        record = build_run_record(repo_root=tmp_root, argv=["--foo", "bar"], args={"x": 1}, dataset_root="/data")

        self.assertIn("versions", record)
        self.assertIn("git", record)
        self.assertEqual(record["argv"], ["--foo", "bar"])
        self.assertEqual(record["args"]["x"], 1)
        self.assertEqual(record["dataset_root"], "/data")
        # Outside git repo should be empty (or at least not crash)
        self.assertIsInstance(record["git"], dict)

    def test_git_info_no_crash(self):
        from yolozu.run_record import git_info

        info = git_info("/tmp")
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
