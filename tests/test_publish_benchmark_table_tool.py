import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPublishBenchmarkTableTool(unittest.TestCase):
    def test_publish_benchmark_table_generates_traceable_outputs(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "publish_benchmark_table.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            report_a = root / "bench_a.json"
            report_b = root / "bench_b.json"

            payload_a = {
                "schema_version": 1,
                "timestamp": "2026-02-19T00:00:00Z",
                "metrics": {
                    "buckets": [
                        {"name": "yolo26n", "metrics": {"fps": 40.0, "latency_ms": {"mean": 25.0}}},
                        {"name": "yolo26s", "metrics": {"fps": 30.0, "latency_ms": {"mean": 33.0}}},
                    ]
                },
                "meta": {"run_id": "run-a", "git_head": "abc"},
            }
            payload_b = {
                "schema_version": 1,
                "timestamp": "2026-02-19T00:01:00Z",
                "metrics": {"summary": {"fps_mean": 20.0, "latency_ms_mean": 50.0}},
                "meta": {"run_id": "run-b", "git_head": "def"},
            }

            report_a.write_text(json.dumps(payload_a), encoding="utf-8")
            report_b.write_text(json.dumps(payload_b), encoding="utf-8")

            out_json = root / "benchmark_table.json"
            out_md = root / "benchmark_table.md"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--report",
                    str(report_a.relative_to(repo_root)),
                    "--report",
                    str(report_b.relative_to(repo_root)),
                    "--output-json",
                    str(out_json.relative_to(repo_root)),
                    "--output-md",
                    str(out_md.relative_to(repo_root)),
                    "--source-command",
                    "python3 tools/benchmark_latency.py --config configs/benchmark_latency_example.json",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"publish_benchmark_table.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            self.assertTrue(out_json.is_file())
            self.assertTrue(out_md.is_file())

            table = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(table.get("kind"), "benchmark_publication_table")
            rows = table.get("rows") or []
            self.assertGreaterEqual(len(rows), 3)
            run_ids = {str(r.get("run_id")) for r in rows}
            self.assertIn("run-a", run_ids)
            self.assertIn("run-b", run_ids)

            md = out_md.read_text(encoding="utf-8")
            self.assertIn("Official benchmark table", md)
            self.assertIn("run-a", md)
            self.assertIn("run-b", md)
            self.assertEqual(table.get("protocol", {}).get("id"), "yolo26")


if __name__ == "__main__":
    unittest.main()
