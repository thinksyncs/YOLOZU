import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _load_train_minimal_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"
    spec = importlib.util.spec_from_file_location("rtdetr_pose_tools_train_minimal", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipIf(torch is None, "torch not installed")
class TestTrainMinimalConfigAndCheckpoint(unittest.TestCase):
    def test_config_sets_defaults_and_cli_overrides(self):
        import tempfile

        mod = _load_train_minimal_module()

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.yaml"
            cfg_path.write_text(
                "batch_size: 7\n"
                "image_size: 123\n"
                "shuffle: false\n"
                "real_images: true\n",
                encoding="utf-8",
            )

            args = mod.parse_args(["--config", str(cfg_path), "--epochs", "2"])
            self.assertEqual(args.batch_size, 7)
            self.assertEqual(args.image_size, 123)
            self.assertEqual(args.shuffle, False)
            self.assertEqual(args.real_images, True)
            self.assertEqual(args.epochs, 2)

    def test_checkpoint_bundle_roundtrip(self):
        import tempfile

        mod = _load_train_minimal_module()

        model = torch.nn.Linear(4, 3)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Do a tiny step so optimizer state is non-trivial.
        x = torch.randn(2, 4)
        y = model(x).sum()
        y.backward()
        optim.step()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "bundle.pt"
            args = mod.parse_args([])
            mod.save_checkpoint_bundle(
                ckpt,
                model=model,
                optim=optim,
                args=args,
                epoch=0,
                global_step=5,
                last_epoch_steps=1,
                last_epoch_avg=1.23,
                last_loss_dict={"loss": torch.tensor(0.5)},
            )

            model2 = torch.nn.Linear(4, 3)
            optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            meta = mod.load_checkpoint_into(model2, optim2, ckpt)

            self.assertEqual(int(meta.get("global_step")), 5)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, model2.state_dict()[k]))


if __name__ == "__main__":
    unittest.main()
