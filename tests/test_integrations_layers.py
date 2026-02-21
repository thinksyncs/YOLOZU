import tempfile
import unittest
from pathlib import Path

from yolozu.integrations.layers.api import run_cli_tool
from yolozu.integrations.layers.core import fail_response, ok_response
from yolozu.integrations.layers.jobs import JobManager
from yolozu.integrations.tool_runner import jobs_list, jobs_status, runs_describe, runs_list


class TestIntegrationLayers(unittest.TestCase):
    def test_core_response_shapes(self):
        ok = ok_response("x", data={"a": 1})
        self.assertTrue(ok["ok"])
        self.assertEqual(ok["tool"], "x")
        self.assertIn("summary", ok)

        err = fail_response("y", message="boom")
        self.assertFalse(err["ok"])
        self.assertEqual(err["tool"], "y")
        self.assertEqual(err["error"], "boom")

    def test_api_layer_blocks_path_traversal(self):
        out = run_cli_tool("validate_predictions", ["validate", "predictions", "../bad.json"])
        self.assertFalse(out["ok"])
        self.assertIn("path traversal", out.get("error", ""))

    def test_jobs_manager_submit_and_status(self):
        manager = JobManager(max_workers=1)
        job_id = manager.submit("dummy", lambda: {"ok": True, "tool": "dummy"})
        status = manager.status(job_id)
        self.assertIsNotNone(status)
        self.assertIn(status["status"], ("queued", "running", "completed"))

    def test_runs_list_and_describe_shape(self):
        runs = runs_list(limit=2)
        self.assertTrue(runs["ok"])
        self.assertIn("runs", runs)

        with tempfile.TemporaryDirectory() as td:
            run_id = Path(td).name
            missing = runs_describe(run_id)
            self.assertIn("ok", missing)

    def test_jobs_api_shape(self):
        lst = jobs_list()
        self.assertTrue(lst["ok"])
        unknown = jobs_status("job_unknown")
        self.assertFalse(unknown["ok"])


if __name__ == "__main__":
    unittest.main()
