import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _write_tiny_rtdetr_pose_config(path: Path) -> None:
    cfg = {
        "dataset": {"root": ".", "split": "val2017", "format": "yolo"},
        "model": {
            "num_classes": 3,
            "hidden_dim": 64,
            "num_queries": 10,
            "use_uncertainty": False,
            "backbone_name": "tiny_cnn",
            "stem_channels": 8,
            "backbone_channels": [16, 32, 64],
            "stage_blocks": [1, 1, 1],
            "num_encoder_layers": 0,
            "num_decoder_layers": 1,
            "nhead": 8,
        },
        "loss": {"name": "default", "task_aligner": "none", "weights": {}},
        "train": {"batch_size": 1, "lr": 0.0001, "epochs": 1},
    }
    path.write_text(json.dumps(cfg))


@unittest.skipIf(torch is None, "torch not installed")
class TestRTDETRPoseBackendSuiteTool(unittest.TestCase):
    def test_torch_only_runs_and_writes_report(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "rtdetr_pose_backend_suite.py"
        self.assertTrue(script.is_file(), "missing tools/rtdetr_pose_backend_suite.py")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            cfg_path = root / "tiny_rtdetr_pose.json"
            _write_tiny_rtdetr_pose_config(cfg_path)

            out_path = root / "suite.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(cfg_path),
                    "--backends",
                    "torch",
                    "--device",
                    "cpu",
                    "--image-size",
                    "32",
                    "--batch",
                    "1",
                    "--samples",
                    "1",
                    "--warmup",
                    "1",
                    "--iterations",
                    "2",
                    "--output",
                    str(out_path),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"rtdetr_pose_backend_suite.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text())
            self.assertIn("parity", payload)
            self.assertIn("benchmark", payload)
            results = payload.get("benchmark", {}).get("results") or []
            self.assertTrue(any(r.get("name") == "torch" for r in results))


if __name__ == "__main__":
    unittest.main()

