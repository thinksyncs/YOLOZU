import unittest
from pathlib import Path


class TestRunRecord(unittest.TestCase):
    def test_build_run_record_outside_git(self):
        from yolozu.run_record import build_run_record, validate_run_record_contract

        tmp_root = Path("/tmp") / "yolozu_run_record_test_no_git"
        # Intentionally do not create a .git dir.
        record = build_run_record(
            repo_root=tmp_root,
            argv=["--foo", "bar"],
            args={"x": 1, "image_size": 64},
            dataset_root="/data",
        )

        self.assertIn("versions", record)
        self.assertIn("git", record)
        self.assertEqual(record["argv"], ["--foo", "bar"])
        self.assertEqual(record["args"]["x"], 1)
        self.assertEqual(record["dataset_root"], "/data")
        self.assertIn("schema_version", record)
        self.assertIn("dependency_lock", record)
        self.assertIn("preprocess", record)
        self.assertIn("command", record)
        self.assertIn("runtime", record)
        self.assertIn("hardware", record)
        self.assertIsInstance(record["git"], dict)
        validate_run_record_contract(record, require_git_sha=False)

    def test_validate_run_record_contract_rejects_missing_preprocess(self):
        from yolozu.run_record import build_run_record, validate_run_record_contract

        record = build_run_record(repo_root=Path("."), argv=["--foo"], args={"x": 1}, dataset_root="/data")
        with self.assertRaises(ValueError):
            validate_run_record_contract(record, require_git_sha=False)

    def test_git_info_no_crash(self):
        from yolozu.run_record import git_info

        info = git_info("/tmp")
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
