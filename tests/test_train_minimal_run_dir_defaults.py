import importlib.util
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "rtdetr_pose"))


def _load_train_minimal_module():
    script_path = repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"
    spec = importlib.util.spec_from_file_location("rtdetr_pose_tools_train_minimal", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTrainMinimalRunDirDefaults(unittest.TestCase):
    def test_plan_accumulation_windows(self):
        mod = _load_train_minimal_module()
        self.assertEqual(mod.plan_accumulation_windows(max_micro_steps=4, grad_accum=2), [2, 2])
        self.assertEqual(mod.plan_accumulation_windows(max_micro_steps=5, grad_accum=2), [2, 2, 1])
        self.assertEqual(mod.plan_accumulation_windows(max_micro_steps=2, grad_accum=4), [2])

    def test_run_dir_populates_default_outputs(self):
        import tempfile

        mod = _load_train_minimal_module()
        with tempfile.TemporaryDirectory() as td:
            args = mod.parse_args(["--run-dir", td])
            args, run_dir = mod.apply_run_dir_defaults(args)
            self.assertIsNotNone(run_dir)
            self.assertTrue(Path(td).exists())

            self.assertEqual(Path(args.metrics_jsonl).name, "metrics.jsonl")
            self.assertEqual(Path(args.metrics_json).name, "metrics.json")
            self.assertEqual(Path(args.metrics_csv).name, "metrics.csv")
            self.assertEqual(Path(args.checkpoint_out).name, "checkpoint.pt")
            self.assertEqual(Path(args.checkpoint_bundle_out).name, "checkpoint_bundle.pt")
            self.assertEqual(Path(args.onnx_out).name, "model.onnx")

    def test_run_dir_does_not_override_explicit_paths(self):
        import tempfile

        mod = _load_train_minimal_module()
        with tempfile.TemporaryDirectory() as td:
            onnx_out = Path(td) / "custom.onnx"
            args = mod.parse_args(["--run-dir", td, "--onnx-out", str(onnx_out)])
            args, _ = mod.apply_run_dir_defaults(args)
            self.assertEqual(Path(args.onnx_out), onnx_out)

    def test_run_contract_sets_default_fracal_stats_out(self):
        import tempfile

        mod = _load_train_minimal_module()
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "dummy.yaml"
            cfg_path.write_text("{}\n", encoding="utf-8")
            args = mod.parse_args(["--run-contract", "--run-id", "unit-test-123", "--config", str(cfg_path)])
            args, contract = mod.apply_run_contract_defaults(args)
            self.assertIsNotNone(contract)
            assert contract is not None
            expected = contract["reports_dir"] / "fracal_stats_bbox.json"
            self.assertEqual(Path(args.fracal_stats_out), expected)

    def test_run_contract_sets_seg_fracal_stats_out_when_task_seg(self):
        import tempfile

        mod = _load_train_minimal_module()
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "dummy.yaml"
            cfg_path.write_text("{}\n", encoding="utf-8")
            args = mod.parse_args(
                [
                    "--run-contract",
                    "--run-id",
                    "unit-test-124",
                    "--config",
                    str(cfg_path),
                    "--fracal-stats-task",
                    "seg",
                ]
            )
            args, contract = mod.apply_run_contract_defaults(args)
            self.assertIsNotNone(contract)
            assert contract is not None
            expected = contract["reports_dir"] / "fracal_stats_seg.json"
            self.assertEqual(Path(args.fracal_stats_out), expected)


if __name__ == "__main__":
    unittest.main()
