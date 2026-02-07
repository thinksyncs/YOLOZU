import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root.parent))

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    DataLoader = None
    Dataset = object

from rtdetr_pose.dataset import build_manifest, extract_pose_intrinsics_targets
from rtdetr_pose.dataset import extract_full_gt_targets, depth_at_bbox_center
from rtdetr_pose.factory import build_losses, build_model
from rtdetr_pose.losses import Losses
from rtdetr_pose.training import build_query_aligned_targets
from rtdetr_pose.model import RTDETRPose

from yolozu.metrics_report import append_jsonl, build_report, write_csv_row, write_json
from yolozu.jitter import default_jitter_profile, sample_intrinsics_jitter, sample_extrinsics_jitter
from yolozu.run_record import build_run_record
from yolozu.sdft import SdftConfig, compute_sdft_loss


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_config_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"config not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise SystemExit("PyYAML is required for YAML configs; install requirements.txt") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}

    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    # Fallback: try JSON then YAML.
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise SystemExit("PyYAML is required for YAML configs; install requirements.txt") from exc
        data = yaml.safe_load(text)
        return data or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal RTDETRPose training scaffold.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML/JSON config file. Values become argparse defaults; explicit CLI flags override.",
    )
    parser.add_argument("--dataset-root", type=str, default="", help="Path to data/coco128")
    parser.add_argument("--split", type=str, default="train2017")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1). Optimizer steps happen every N batches.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        help="Torch device for training (e.g., cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--amp",
        choices=("none", "fp16", "bf16"),
        default="none",
        help="Automatic mixed precision (cuda only): none|fp16|bf16 (default: none).",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DistributedDataParallel (single node). Use via torchrun; outputs written on rank0 only.",
    )
    parser.add_argument(
        "--ddp-backend",
        default=None,
        help="DDP backend override (default: nccl for cuda, else gloo).",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=0.0,
        help="If >0, clip gradients to this max norm before optimizer step.",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Linear warmup steps for learning rate (0 disables warmup).",
    )
    parser.add_argument(
        "--lr-warmup-init",
        type=float,
        default=0.0,
        help="Initial learning rate value at step 0 for warmup.",
    )
    parser.add_argument("--max-steps", type=int, default=30, help="Cap steps per epoch")
    parser.add_argument("--log-every", type=int, default=10, help="Print every N steps")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Enable random multiscale resize around --image-size.",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.8,
        help="Lower bound for multiscale resize (relative to --image-size).",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.2,
        help="Upper bound for multiscale resize (relative to --image-size).",
    )
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.0,
        help="Probability of random horizontal flip augmentation.",
    )
    parser.add_argument(
        "--hflip-prob-start",
        type=float,
        default=None,
        help="Optional starting hflip probability for linear schedule.",
    )
    parser.add_argument(
        "--hflip-prob-end",
        type=float,
        default=None,
        help="Optional ending hflip probability for linear schedule.",
    )
    parser.add_argument(
        "--intrinsics-jitter",
        action="store_true",
        help="Enable intrinsics jitter augmentation on K_gt.",
    )
    parser.add_argument(
        "--sim-jitter",
        action="store_true",
        help="Enable SIM-style intrinsics jitter using yolozu.jitter profiles.",
    )
    parser.add_argument(
        "--sim-jitter-profile",
        type=str,
        default=None,
        help="Optional JSON file to override the default SIM jitter profile.",
    )
    parser.add_argument(
        "--sim-jitter-extrinsics",
        action="store_true",
        help="Apply SIM profile extrinsics jitter to gt_t/gt_R.",
    )
    parser.add_argument(
        "--extrinsics-jitter",
        action="store_true",
        help="Enable manual extrinsics jitter on gt_t/gt_R.",
    )
    parser.add_argument("--jitter-dx", type=float, default=0.01, help="Translation jitter range in meters.")
    parser.add_argument("--jitter-dy", type=float, default=0.01, help="Translation jitter range in meters.")
    parser.add_argument("--jitter-dz", type=float, default=0.02, help="Translation jitter range in meters.")
    parser.add_argument("--jitter-droll", type=float, default=1.0, help="Roll jitter range in degrees.")
    parser.add_argument("--jitter-dpitch", type=float, default=1.0, help="Pitch jitter range in degrees.")
    parser.add_argument("--jitter-dyaw", type=float, default=2.0, help="Yaw jitter range in degrees.")
    parser.add_argument(
        "--jitter-dfx",
        type=float,
        default=0.02,
        help="Relative fx jitter range (uniform in [-dfx, dfx]).",
    )
    parser.add_argument(
        "--jitter-dfy",
        type=float,
        default=0.02,
        help="Relative fy jitter range (uniform in [-dfy, dfy]).",
    )
    parser.add_argument(
        "--jitter-dcx",
        type=float,
        default=4.0,
        help="Absolute cx jitter range in pixels (uniform in [-dcx, dcx]).",
    )
    parser.add_argument(
        "--jitter-dcy",
        type=float,
        default=4.0,
        help="Absolute cy jitter range in pixels (uniform in [-dcy, dcy]).",
    )
    parser.add_argument(
        "--real-images",
        action="store_true",
        help="Load real images via record['image_path'] (requires Pillow). Default uses synthetic images.",
    )
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument(
        "--use-uncertainty",
        action="store_true",
        help="Enable uncertainty heads (log_sigma_z/log_sigma_rot) for task alignment.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Shuffle dataset each epoch (default: true).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic data order via DataLoader generator (seeded).",
    )
    parser.add_argument("--use-matcher", action="store_true", help="Use Hungarian matching")
    parser.add_argument("--cost-cls", type=float, default=1.0)
    parser.add_argument("--cost-bbox", type=float, default=5.0)
    parser.add_argument("--cost-z", type=float, default=0.0, help="Optional matching cost for depth")
    parser.add_argument(
        "--cost-rot",
        type=float,
        default=0.0,
        help="Optional matching cost for rotation (geodesic angle)",
    )
    parser.add_argument(
        "--synthetic-pose",
        action="store_true",
        help="Generate synthetic z/R GT per instance (scaffold only)",
    )
    parser.add_argument(
        "--z-from-dobj",
        action="store_true",
        help="When GT t is missing, derive z (and optionally t if K is available) from D_obj at bbox center",
    )
    parser.add_argument(
        "--load-aux",
        action="store_true",
        help="Allow loading mask/depth arrays from paths (.json/.npy) for z-from-dobj; default keeps lazy paths",
    )
    parser.add_argument(
        "--cost-t",
        type=float,
        default=0.0,
        help="Optional matching cost for translation recovered from (bbox, offsets, z, K')",
    )
    parser.add_argument(
        "--debug-losses",
        action="store_true",
        help="Print loss dict breakdown on step 1",
    )
    parser.add_argument(
        "--task-aligner",
        choices=("none", "uncertainty"),
        default="none",
        help="Multi-task loss alignment strategy (default: none).",
    )
    parser.add_argument("--metrics-jsonl", default=None, help="Append per-step loss/metric report JSONL here.")
    parser.add_argument("--metrics-json", default=None, help="Write final run summary JSON here.")
    parser.add_argument("--metrics-csv", default=None, help="Write final run summary CSV (single row) here.")

    # Continual learning / self-distillation (SDFT-inspired)
    parser.add_argument(
        "--self-distill-from",
        default=None,
        help="Optional teacher checkpoint to distill against (to reduce catastrophic forgetting).",
    )
    parser.add_argument(
        "--self-distill-weight",
        type=float,
        default=1.0,
        help="Global multiplier for self-distillation loss (only used when --self-distill-from is set).",
    )
    parser.add_argument(
        "--self-distill-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for logits distillation (>=1 recommended).",
    )
    parser.add_argument(
        "--self-distill-kl",
        choices=("forward", "reverse", "sym"),
        default="reverse",
        help="KL direction for logits distillation (default: reverse, SDFT-style).",
    )
    parser.add_argument(
        "--self-distill-keys",
        type=str,
        default="logits,bbox",
        help="Comma-separated model output keys to distill (default: logits,bbox).",
    )
    parser.add_argument(
        "--self-distill-logits-weight",
        type=float,
        default=1.0,
        help="Per-key weight for logits distillation term.",
    )
    parser.add_argument(
        "--self-distill-bbox-weight",
        type=float,
        default=1.0,
        help="Per-key weight for bbox distillation term (compared in sigmoid space).",
    )
    parser.add_argument(
        "--self-distill-other-l1-weight",
        type=float,
        default=1.0,
        help="Per-key L1 weight for any other distilled tensor outputs.",
    )

    # Checkpointing / resume
    parser.add_argument("--resume-from", default=None, help="Resume weights or full checkpoint bundle from this path.")
    parser.add_argument(
        "--checkpoint-bundle-out",
        default=None,
        help="Write a full checkpoint bundle (model+optimizer+progress) to this path at end.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If >0 and --checkpoint-bundle-out is set, also save intermediate bundles every N steps.",
    )

    # Back-compat: weights-only checkpoint
    parser.add_argument("--checkpoint-out", default=None, help="Write model state_dict to this path at end.")

    # Reproducible artifacts
    parser.add_argument(
        "--run-dir",
        default=None,
        help="If set, write standard artifacts into this folder (run_record.json, metrics.jsonl/json/csv, checkpoint*.pt, model.onnx).",
    )
    parser.add_argument(
        "--onnx-out",
        default=None,
        help="Optional ONNX export output path (overrides --run-dir default).",
    )
    parser.add_argument(
        "--onnx-meta-out",
        default=None,
        help="Optional ONNX export metadata JSON path (default: <onnx-out>.meta.json).",
    )
    parser.add_argument("--onnx-opset", type=int, default=17, help="ONNX opset version (default: 17).")
    parser.add_argument(
        "--onnx-dynamic-hw",
        action="store_true",
        help="Export ONNX with dynamic height/width axes (batch is always dynamic).",
    )
    return parser


def apply_run_dir_defaults(args: argparse.Namespace) -> tuple[argparse.Namespace, Path | None]:
    run_dir = None
    if args.run_dir:
        run_dir = Path(str(args.run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)

        def _default_path(current: str | None, name: str) -> str:
            return current if current else str(run_dir / name)

        args.metrics_jsonl = _default_path(args.metrics_jsonl, "metrics.jsonl")
        args.metrics_json = _default_path(args.metrics_json, "metrics.json")
        args.metrics_csv = _default_path(args.metrics_csv, "metrics.csv")
        args.checkpoint_out = _default_path(args.checkpoint_out, "checkpoint.pt")
        args.checkpoint_bundle_out = _default_path(args.checkpoint_bundle_out, "checkpoint_bundle.pt")
        args.onnx_out = _default_path(args.onnx_out, "model.onnx")

    return args, run_dir


def unwrap_model(model: "torch.nn.Module") -> "torch.nn.Module":
    if hasattr(model, "module"):
        try:
            return model.module
        except Exception:
            return model
    return model


def collect_torch_cuda_meta() -> dict[str, Any]:
    if torch is None:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda.is_available() is false"}
    try:
        idx = int(torch.cuda.current_device())
    except Exception:
        idx = 0
    meta: dict[str, Any] = {
        "available": True,
        "device_index": idx,
        "device_name": torch.cuda.get_device_name(idx),
        "device_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(idx)),
        "total_memory_mb": int(torch.cuda.get_device_properties(idx).total_memory // (1024 * 1024)),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None),
    }
    return meta


def _rotation_matrix_from_rpy(roll_rad: float, pitch_rad: float, yaw_rad: float) -> "torch.Tensor":
    cr = math.cos(roll_rad)
    sr = math.sin(roll_rad)
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=torch.float32)
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=torch.float32)
    rz = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return rz @ ry @ rx


def compute_warmup_lr(base_lr: float, step: int, warmup_steps: int, warmup_init: float) -> float:
    if warmup_steps <= 0:
        return float(base_lr)
    if step <= 0:
        return float(warmup_init)
    if step >= warmup_steps:
        return float(base_lr)
    alpha = float(step) / float(warmup_steps)
    return float(warmup_init + (base_lr - warmup_init) * alpha)


def compute_linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(end)
    alpha = min(max(float(step) / float(total_steps - 1), 0.0), 1.0)
    return float(start + (end - start) * alpha)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = build_parser()

    # Two-stage parse so config can set defaults.
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        cfg = load_config_file(pre.config)
        if not isinstance(cfg, dict):
            raise SystemExit(f"config must be a dict/object at top-level: {pre.config}")
        # Only apply keys that argparse knows.
        known = {a.dest for a in parser._actions if getattr(a, "dest", None)}
        defaults: dict[str, Any] = {}
        for key, value in cfg.items():
            if key in known:
                defaults[key] = value
        parser.set_defaults(**defaults)

    return parser.parse_args(argv)


def load_checkpoint_into(model: "torch.nn.Module", optim: "torch.optim.Optimizer | None", path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    meta: dict[str, Any] = {"path": str(path)}

    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        if optim is not None and isinstance(obj.get("optim_state_dict"), dict):
            try:
                optim.load_state_dict(obj["optim_state_dict"])
                meta["optim_loaded"] = True
            except Exception:
                meta["optim_loaded"] = False
        meta.update({k: obj.get(k) for k in ("epoch", "global_step") if k in obj})
        return meta

    # Assume weights-only state_dict
    if isinstance(obj, dict):
        model.load_state_dict(obj)
        meta["optim_loaded"] = False
        return meta

    raise SystemExit(f"unrecognized checkpoint format: {path}")


def save_checkpoint_bundle(
    path: str | Path,
    *,
    model: "torch.nn.Module",
    optim: "torch.optim.Optimizer | None",
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    last_epoch_steps: int,
    last_epoch_avg: float | None,
    last_loss_dict: dict[str, Any] | None,
    run_record: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "last_epoch_steps": int(last_epoch_steps),
        "last_epoch_avg": float(last_epoch_avg) if last_epoch_avg is not None else None,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "run_record": run_record
        if run_record is not None
        else build_run_record(
            repo_root=repo_root.parent,
            args=vars(args),
            dataset_root=(getattr(args, "dataset_root", "") or None),
        ),
    }
    if optim is not None:
        payload["optim_state_dict"] = optim.state_dict()
    if last_loss_dict is not None:
        payload["last_loss"] = {
            k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")
        }
    torch.save(payload, path)


class ManifestDataset(Dataset):
    def __init__(
        self,
        records,
        num_queries,
        num_classes,
        image_size,
        seed,
        use_matcher,
        synthetic_pose,
        z_from_dobj,
        load_aux,
        real_images,
        multiscale,
        scale_min,
        scale_max,
        hflip_prob,
        intrinsics_jitter,
        jitter_dfx,
        jitter_dfy,
        jitter_dcx,
        jitter_dcy,
        sim_jitter,
        sim_jitter_profile,
        sim_jitter_extrinsics,
        extrinsics_jitter,
        jitter_dx,
        jitter_dy,
        jitter_dz,
        jitter_droll,
        jitter_dpitch,
        jitter_dyaw,
    ):
        self.records = records
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.use_matcher = bool(use_matcher)
        self.synthetic_pose = bool(synthetic_pose)
        self.z_from_dobj = bool(z_from_dobj)
        self.load_aux = bool(load_aux)
        self.real_images = bool(real_images)
        self.multiscale = bool(multiscale)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.hflip_prob = float(hflip_prob)
        self.intrinsics_jitter = bool(intrinsics_jitter)
        self.jitter_dfx = float(jitter_dfx)
        self.jitter_dfy = float(jitter_dfy)
        self.jitter_dcx = float(jitter_dcx)
        self.jitter_dcy = float(jitter_dcy)
        self.sim_jitter = bool(sim_jitter)
        self.sim_jitter_profile = sim_jitter_profile
        self.sim_jitter_extrinsics = bool(sim_jitter_extrinsics)
        self.extrinsics_jitter = bool(extrinsics_jitter)
        self.jitter_dx = float(jitter_dx)
        self.jitter_dy = float(jitter_dy)
        self.jitter_dz = float(jitter_dz)
        self.jitter_droll = float(jitter_droll)
        self.jitter_dpitch = float(jitter_dpitch)
        self.jitter_dyaw = float(jitter_dyaw)

    def _load_rgb_image_tensor(self, image_path: Path, target_size: int) -> "torch.Tensor | None":
        if not image_path.exists():
            return None
        try:
            from PIL import Image
        except Exception as exc:
            raise SystemExit(
                "Pillow is required for --real-images. Install it (e.g. pip install Pillow) or omit --real-images."
            ) from exc
        try:
            import numpy as np
        except Exception as exc:
            raise SystemExit(
                "NumPy is required for --real-images. Install it (e.g. pip install numpy) or omit --real-images."
            ) from exc

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return None

        if target_size > 0:
            img = img.resize((target_size, target_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        arr = arr / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_2d(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value
        if not self.load_aux:
            return None
        if isinstance(value, str):
            path = Path(value)
            if path.suffix.lower() == ".json":
                import json

                try:
                    return json.loads(path.read_text())
                except Exception:
                    return None
            if path.suffix.lower() in (".npy", ".npz"):
                try:
                    import numpy as np
                except Exception:
                    return None
                try:
                    loaded = np.load(path)
                except Exception:
                    return None
                if hasattr(loaded, "files"):
                    if not loaded.files:
                        return None
                    return loaded[loaded.files[0]]
                return loaded
        return None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        gen = torch.Generator()
        gen.manual_seed(self.seed + int(idx))

        base_size = max(1, int(self.image_size))
        scale = 1.0
        if self.multiscale:
            lo = min(self.scale_min, self.scale_max)
            hi = max(self.scale_min, self.scale_max)
            if hi > 0 and lo > 0:
                scale = float(torch.rand((), generator=gen) * (hi - lo) + lo)
        target_size = max(1, int(round(base_size * scale)))
        scale_factor = float(target_size) / float(base_size)

        flip = False
        if self.hflip_prob > 0:
            flip = bool(torch.rand((), generator=gen) < float(self.hflip_prob))

        # This is a training-loop scaffold.
        # By default we use synthetic images (keeps deps minimal), but an optional
        # mode can load real JPEGs from record['image_path'].
        image = None
        if self.real_images:
            image_path_raw = record.get("image_path")
            if image_path_raw:
                image = self._load_rgb_image_tensor(Path(str(image_path_raw)), target_size)
        if image is None:
            image = torch.rand(3, target_size, target_size, generator=gen)

        if image.shape[-1] != target_size or image.shape[-2] != target_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if flip:
            image = torch.flip(image, dims=(2,))

        instances = record.get("labels") or []
        if self.use_matcher:
            full = extract_full_gt_targets(record, num_instances=len(instances))
            gt_labels = []
            gt_bbox = []
            gt_z = []
            gt_z_mask = []
            gt_R = []
            gt_R_mask = []
            gt_t = []
            gt_t_mask = []
            gt_offsets = []
            gt_offsets_mask = []
            gt_M_mask = []
            gt_D_obj_mask = []
            gt_M = []
            gt_D_obj = []

            # We don't decode images, so default HW is the generated tensor size.
            image_hw = torch.tensor([float(target_size), float(target_size)], dtype=torch.float32)

            # Prefer real intrinsics when present; else synthesize if requested.
            K_gt = None
            if full.get("K_gt") is not None:
                K_gt = torch.tensor(full["K_gt"], dtype=torch.float32)
            elif self.synthetic_pose:
                w = float(self.image_size)
                h = float(self.image_size)
                fx = w
                fy = w
                cx = w * 0.5
                cy = h * 0.5
                K_gt = torch.tensor(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                )

            if K_gt is not None and scale_factor != 1.0:
                K_gt = K_gt.clone()
                K_gt[0, 0] *= scale_factor
                K_gt[1, 1] *= scale_factor
                K_gt[0, 2] *= scale_factor
                K_gt[1, 2] *= scale_factor
            if K_gt is not None and flip:
                K_gt = K_gt.clone()
                K_gt[0, 2] = float(target_size - 1.0) - K_gt[0, 2]
            if K_gt is not None and self.sim_jitter and self.sim_jitter_profile:
                jitter = sample_intrinsics_jitter(self.sim_jitter_profile, seed=self.seed + int(idx))
                K_gt = K_gt.clone()
                K_gt[0, 0] = K_gt[0, 0] * (1.0 + float(jitter.get("dfx", 0.0)))
                K_gt[1, 1] = K_gt[1, 1] * (1.0 + float(jitter.get("dfy", 0.0)))
                K_gt[0, 2] = K_gt[0, 2] + float(jitter.get("dcx", 0.0))
                K_gt[1, 2] = K_gt[1, 2] + float(jitter.get("dcy", 0.0))
            elif K_gt is not None and self.intrinsics_jitter:
                K_gt = K_gt.clone()
                dfx = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dfx)
                dfy = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dfy)
                dcx = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dcx)
                dcy = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dcy)
                K_gt[0, 0] = K_gt[0, 0] * (1.0 + dfx)
                K_gt[1, 1] = K_gt[1, 1] * (1.0 + dfy)
                K_gt[0, 2] = K_gt[0, 2] + dcx
                K_gt[1, 2] = K_gt[1, 2] + dcy

            for inst_i, inst in enumerate(instances):
                class_id = int(inst.get("class_id", -1))
                if not (0 <= class_id < self.num_classes):
                    continue
                bb = inst.get("bbox") or {}
                cx = float(bb.get("cx", 0.0))
                cy = float(bb.get("cy", 0.0))
                w = float(bb.get("w", 0.0))
                h = float(bb.get("h", 0.0))
                if flip:
                    cx = 1.0 - cx
                gt_labels.append(class_id)
                gt_bbox.append([cx, cy, w, h])

                # Real (t/R/offsets) if present, otherwise optional synthetic fallback.
                t_i = full.get("t_gt", [None])[inst_i] if full.get("t_gt") is not None else None
                r_i = full.get("R_gt", [None])[inst_i] if full.get("R_gt") is not None else None
                off_i = full.get("offsets_gt", [None])[inst_i] if full.get("offsets_gt") is not None else None

                m_i = full.get("M", [None])[inst_i] if full.get("M") is not None else None
                d_i = full.get("D_obj", [None])[inst_i] if full.get("D_obj") is not None else None
                m_loaded = self._load_2d(m_i)
                d_loaded = self._load_2d(d_i)
                gt_M_mask.append(bool(full.get("M_mask", [False])[inst_i]) if full.get("M_mask") else False)
                gt_D_obj_mask.append(
                    bool(full.get("D_obj_mask", [False])[inst_i]) if full.get("D_obj_mask") else False
                )
                gt_M.append(m_loaded)
                gt_D_obj.append(d_loaded)

                # z/t
                if t_i is not None:
                    t_val = [float(v) for v in t_i]
                    z_val = float(t_val[2])
                    gt_t.append(t_val)
                    gt_t_mask.append(True)
                    gt_z.append(z_val)
                    gt_z_mask.append(True)
                elif self.synthetic_pose:
                    z_val = float(torch.rand((), generator=gen) * 0.9 + 0.1)
                    gt_z.append(z_val)
                    gt_z_mask.append(True)
                    if K_gt is None:
                        raise RuntimeError("synthetic_pose requires K_gt")
                    cx_n = float(bb.get("cx", 0.0))
                    cy_n = float(bb.get("cy", 0.0))
                    u = cx_n * float(image_hw[1])
                    v = cy_n * float(image_hw[0])
                    x = (u - float(K_gt[0, 2])) / float(K_gt[0, 0]) * z_val
                    y = (v - float(K_gt[1, 2])) / float(K_gt[1, 1]) * z_val
                    gt_t.append([x, y, z_val])
                    gt_t_mask.append(True)
                else:
                    # Optional: derive z (and optionally t) from D_obj at bbox center.
                    z_val = None
                    if self.z_from_dobj and d_loaded is not None:
                        z_val = depth_at_bbox_center(d_loaded, bb, mask=m_loaded)
                    if z_val is not None:
                        gt_z.append(float(z_val))
                        gt_z_mask.append(True)
                        if K_gt is not None:
                            cx_n = float(bb.get("cx", 0.0))
                            cy_n = float(bb.get("cy", 0.0))
                            u = cx_n * float(image_hw[1])
                            v = cy_n * float(image_hw[0])
                            x = (u - float(K_gt[0, 2])) / float(K_gt[0, 0]) * float(z_val)
                            y = (v - float(K_gt[1, 2])) / float(K_gt[1, 1]) * float(z_val)
                            gt_t.append([x, y, float(z_val)])
                            gt_t_mask.append(True)
                        else:
                            gt_t.append([0.0, 0.0, float(z_val)])
                            gt_t_mask.append(False)
                    else:
                        gt_z.append(0.0)
                        gt_z_mask.append(False)
                        gt_t.append([0.0, 0.0, 0.0])
                        gt_t_mask.append(False)

                # R
                if r_i is not None:
                    gt_R.append(torch.tensor(r_i, dtype=torch.float32))
                    gt_R_mask.append(True)
                elif self.synthetic_pose:
                    a = torch.randn(3, 3, generator=gen)
                    q, _ = torch.linalg.qr(a)
                    if torch.det(q) < 0:
                        q[:, 0] = -q[:, 0]
                    gt_R.append(q)
                    gt_R_mask.append(True)
                else:
                    gt_R.append(torch.eye(3, dtype=torch.float32))
                    gt_R_mask.append(False)

                # offsets
                if off_i is not None:
                    du = float(off_i[0]) * scale_factor
                    dv = float(off_i[1]) * scale_factor
                    if flip:
                        du = -du
                    gt_offsets.append([du, dv])
                    gt_offsets_mask.append(True)
                elif self.synthetic_pose:
                    gt_offsets.append([0.0, 0.0])
                    gt_offsets_mask.append(True)
                else:
                    gt_offsets.append([0.0, 0.0])
                    gt_offsets_mask.append(False)

            if (self.sim_jitter and self.sim_jitter_profile and self.sim_jitter_extrinsics) or self.extrinsics_jitter:
                if self.sim_jitter and self.sim_jitter_profile and self.sim_jitter_extrinsics:
                    jitter = sample_extrinsics_jitter(self.sim_jitter_profile, seed=self.seed + int(idx))
                    dx = float(jitter.get("dx", 0.0))
                    dy = float(jitter.get("dy", 0.0))
                    dz = float(jitter.get("dz", 0.0))
                    droll = float(jitter.get("droll", 0.0))
                    dpitch = float(jitter.get("dpitch", 0.0))
                    dyaw = float(jitter.get("dyaw", 0.0))
                else:
                    dx = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dx)
                    dy = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dy)
                    dz = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dz)
                    droll = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_droll)
                    dpitch = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dpitch)
                    dyaw = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dyaw)

                for j in range(len(gt_t)):
                    if gt_t_mask[j]:
                        gt_t[j] = [gt_t[j][0] + dx, gt_t[j][1] + dy, gt_t[j][2] + dz]
                if any(gt_R_mask):
                    r_delta = _rotation_matrix_from_rpy(
                        math.radians(droll),
                        math.radians(dpitch),
                        math.radians(dyaw),
                    )
                    gt_R = [r_delta @ r if mask else r for r, mask in zip(gt_R, gt_R_mask)]

            num_inst = len(gt_labels)
            m_tensor = None
            d_tensor = None
            if self.load_aux and num_inst > 0:
                for item in gt_M:
                    if item is not None:
                        m_tensor = torch.as_tensor(item, dtype=torch.float32)
                        break
                if m_tensor is not None:
                    m_hw = tuple(m_tensor.shape)
                    stacked = []
                    for item in gt_M:
                        if item is None:
                            stacked.append(torch.zeros(m_hw, dtype=torch.float32))
                            continue
                        t = torch.as_tensor(item, dtype=torch.float32)
                        if tuple(t.shape) != m_hw:
                            stacked.append(torch.zeros(m_hw, dtype=torch.float32))
                        else:
                            stacked.append(t)
                    m_tensor = torch.stack(stacked, dim=0)

                for item in gt_D_obj:
                    if item is not None:
                        d_tensor = torch.as_tensor(item, dtype=torch.float32)
                        break
                if d_tensor is not None:
                    d_hw = tuple(d_tensor.shape)
                    stacked = []
                    for item in gt_D_obj:
                        if item is None:
                            stacked.append(torch.zeros(d_hw, dtype=torch.float32))
                            continue
                        t = torch.as_tensor(item, dtype=torch.float32)
                        if tuple(t.shape) != d_hw:
                            stacked.append(torch.zeros(d_hw, dtype=torch.float32))
                        else:
                            stacked.append(t)
                    d_tensor = torch.stack(stacked, dim=0)
            if num_inst == 0:
                gt_labels_t = torch.empty((0,), dtype=torch.long)
                gt_bbox_t = torch.empty((0, 4), dtype=torch.float32)
                gt_z_t = torch.empty((0, 1), dtype=torch.float32)
                gt_z_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_R_t = torch.empty((0, 3, 3), dtype=torch.float32)
                gt_R_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_t_t = torch.empty((0, 3), dtype=torch.float32)
                gt_t_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_offsets_t = torch.empty((0, 2), dtype=torch.float32)
                gt_offsets_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_M_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_D_obj_mask_t = torch.empty((0,), dtype=torch.bool)
            else:
                gt_labels_t = torch.tensor(gt_labels, dtype=torch.long)
                gt_bbox_t = torch.tensor(gt_bbox, dtype=torch.float32)
                gt_z_t = torch.tensor(gt_z, dtype=torch.float32).unsqueeze(-1)
                gt_z_mask_t = torch.tensor(gt_z_mask, dtype=torch.bool)
                gt_R_t = torch.stack(gt_R, dim=0)
                gt_R_mask_t = torch.tensor(gt_R_mask, dtype=torch.bool)
                gt_t_t = torch.tensor(gt_t, dtype=torch.float32)
                gt_t_mask_t = torch.tensor(gt_t_mask, dtype=torch.bool)
                gt_offsets_t = torch.tensor(gt_offsets, dtype=torch.float32)
                gt_offsets_mask_t = torch.tensor(gt_offsets_mask, dtype=torch.bool)
                gt_M_mask_t = torch.tensor(gt_M_mask, dtype=torch.bool)
                gt_D_obj_mask_t = torch.tensor(gt_D_obj_mask, dtype=torch.bool)

            return {
                "image": image,
                "targets": {
                    "gt_labels": gt_labels_t,
                    "gt_bbox": gt_bbox_t,
                    "gt_z": gt_z_t,
                    "gt_z_mask": gt_z_mask_t,
                    "gt_R": gt_R_t,
                    "gt_R_mask": gt_R_mask_t,
                    "gt_t": gt_t_t,
                    "gt_t_mask": gt_t_mask_t,
                    "gt_offsets": gt_offsets_t,
                    "gt_offsets_mask": gt_offsets_mask_t,
                    "gt_M_mask": gt_M_mask_t,
                    "gt_D_obj_mask": gt_D_obj_mask_t,
                    **({"gt_M": m_tensor} if m_tensor is not None else {}),
                    **({"gt_D_obj": d_tensor} if d_tensor is not None else {}),
                    **({"K_gt": K_gt} if K_gt is not None else {}),
                    "image_hw": image_hw,
                },
            }

        labels = torch.full((self.num_queries,), -1, dtype=torch.long)
        bbox = torch.zeros((self.num_queries, 4), dtype=torch.float32)
        for qi, inst in enumerate(instances[: self.num_queries]):
            class_id = int(inst.get("class_id", -1))
            if 0 <= class_id < self.num_classes:
                labels[qi] = class_id
            bb = inst.get("bbox") or {}
            cx = float(bb.get("cx", 0.0))
            if flip:
                cx = 1.0 - cx
            bbox[qi, 0] = cx
            bbox[qi, 1] = float(bb.get("cy", 0.0))
            bbox[qi, 2] = float(bb.get("w", 0.0))
            bbox[qi, 3] = float(bb.get("h", 0.0))

        return {"image": image, "targets": {"labels": labels, "bbox": bbox}}


def _pad_field(targets, key, max_len, *, pad_value=0.0, dtype=None):
    if max_len == 0:
        return None
    tail = None
    for tgt in targets:
        value = tgt.get(key)
        if isinstance(value, torch.Tensor):
            if dtype is None:
                dtype = value.dtype
            tail = value.shape[1:]
            break
    if dtype is None:
        return None
    if tail is None:
        tail = ()

    rows = []
    for tgt in targets:
        value = tgt.get(key)
        if not isinstance(value, torch.Tensor):
            value = torch.empty((0, *tail), dtype=dtype)
        pad_len = max_len - value.shape[0]
        if pad_len < 0:
            raise ValueError(f"{key} has more instances than max_len")
        if pad_len == 0:
            padded = value
        else:
            pad = torch.full((pad_len, *tail), pad_value, dtype=dtype)
            padded = torch.cat([value, pad], dim=0)
        rows.append(padded)
    return torch.stack(rows, dim=0)


def collate(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [item["targets"] for item in batch]
    if not targets or "gt_labels" not in targets[0]:
        return images, targets

    counts = torch.tensor(
        [int(tgt.get("gt_labels").shape[0]) if isinstance(tgt.get("gt_labels"), torch.Tensor) else 0 for tgt in targets],
        dtype=torch.long,
    )
    max_len = int(counts.max().item()) if counts.numel() else 0
    if max_len == 0:
        padded = {
            "gt_count": counts,
            "gt_mask": torch.zeros((len(targets), 0), dtype=torch.bool),
        }
        return images, {"per_sample": targets, "padded": padded}

    padded = {
        "gt_count": counts,
        "gt_mask": (torch.arange(max_len).unsqueeze(0) < counts.unsqueeze(1)),
        "gt_labels": _pad_field(targets, "gt_labels", max_len, pad_value=-1, dtype=torch.long),
        "gt_bbox": _pad_field(targets, "gt_bbox", max_len, pad_value=0.0, dtype=torch.float32),
    }

    optional_fields = [
        ("gt_z", 0.0, torch.float32),
        ("gt_z_mask", False, torch.bool),
        ("gt_R", 0.0, torch.float32),
        ("gt_R_mask", False, torch.bool),
        ("gt_t", 0.0, torch.float32),
        ("gt_t_mask", False, torch.bool),
        ("gt_offsets", 0.0, torch.float32),
        ("gt_offsets_mask", False, torch.bool),
        ("gt_M_mask", False, torch.bool),
        ("gt_D_obj_mask", False, torch.bool),
        ("gt_M", 0.0, torch.float32),
        ("gt_D_obj", 0.0, torch.float32),
    ]
    for key, pad_value, dtype in optional_fields:
        if any(key in tgt for tgt in targets):
            padded[key] = _pad_field(targets, key, max_len, pad_value=pad_value, dtype=dtype)

    return images, {"per_sample": targets, "padded": padded}


def main(argv: list[str] | None = None) -> int:
    if torch is None:  # pragma: no cover
        raise SystemExit("torch is required; install requirements-test.txt")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args, run_dir = apply_run_dir_defaults(args)

    # Optional DDP (torchrun sets WORLD_SIZE/RANK/LOCAL_RANK)
    world_size_env = int(os.environ.get("WORLD_SIZE", "1") or "1")
    ddp_enabled = bool(args.ddp) or world_size_env > 1
    if bool(args.ddp) and world_size_env <= 1:
        raise SystemExit("--ddp requires torchrun (WORLD_SIZE>1). Example: torchrun --nproc_per_node=2 ... --ddp")
    rank = int(os.environ.get("RANK", "0") or "0") if ddp_enabled else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0") if ddp_enabled else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1") if ddp_enabled else 1
    if ddp_enabled:
        backend = str(args.ddp_backend or ("nccl" if torch.cuda.is_available() else "gloo"))
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    is_main = rank == 0

    sim_profile = None
    if args.sim_jitter:
        sim_profile = default_jitter_profile()
        if args.sim_jitter_profile:
            path = Path(args.sim_jitter_profile)
            if not path.exists():
                raise SystemExit(f"sim jitter profile not found: {path}")
            try:
                sim_profile = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise SystemExit(f"failed to load sim jitter profile: {path}") from exc

    run_record = build_run_record(
        repo_root=repo_root.parent,
        argv=(sys.argv[1:] if argv is None else argv),
        args=vars(args),
        dataset_root=(args.dataset_root or None),
        extra={
            "timestamp_utc": _now_utc(),
            "ddp": {"enabled": bool(ddp_enabled), "backend": (str(args.ddp_backend) if args.ddp_backend else None), "rank": rank, "local_rank": local_rank, "world_size": world_size},
            "cuda": collect_torch_cuda_meta(),
        },
    )

    torch.manual_seed(args.seed)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = repo_root / "data" / "coco128"
        if not dataset_root.exists():
            dataset_root = repo_root.parent / "data" / "coco128"

    model_cfg = None
    loss_cfg = None
    if args.config:
        try:
            from rtdetr_pose.config import load_config
        except Exception:
            load_config = None
        if load_config is not None:
            try:
                cfg_obj = load_config(args.config)
                model_cfg = cfg_obj.model
                loss_cfg = getattr(cfg_obj, "loss", None)
            except Exception:
                model_cfg = None
                loss_cfg = None
    if model_cfg is not None:
        args.num_queries = int(model_cfg.num_queries)
        args.num_classes = int(model_cfg.num_classes)

    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest.get("images") or []
    if not records:
        raise SystemExit(
            f"No records found under {dataset_root}. "
            "Fetch coco128 first: bash tools/fetch_coco128.sh"
        )

    if is_main:
        stats = {
            "mask": 0,
            "depth": 0,
            "pose": 0,
            "intrinsics": 0,
            "cad_points": 0,
        }
        for rec in records:
            if rec.get("mask_path") is not None:
                stats["mask"] += 1
            if rec.get("depth_path") is not None:
                stats["depth"] += 1
            if rec.get("R_gt") is not None or rec.get("t_gt") is not None or rec.get("pose") is not None:
                stats["pose"] += 1
            if rec.get("K_gt") is not None or rec.get("intrinsics") is not None:
                stats["intrinsics"] += 1
            if rec.get("cad_points") is not None:
                stats["cad_points"] += 1
        print(
            "dataset_stats "
            + " ".join(f"{key}={value}" for key, value in sorted(stats.items()))
        )

    ds = ManifestDataset(
        records,
        num_queries=args.num_queries,
        num_classes=args.num_classes,
        image_size=args.image_size,
        seed=args.seed,
        use_matcher=args.use_matcher,
        synthetic_pose=args.synthetic_pose,
        z_from_dobj=args.z_from_dobj,
        load_aux=args.load_aux,
        real_images=args.real_images,
        multiscale=args.multiscale,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        hflip_prob=args.hflip_prob,
        intrinsics_jitter=args.intrinsics_jitter,
        jitter_dfx=args.jitter_dfx,
        jitter_dfy=args.jitter_dfy,
        jitter_dcx=args.jitter_dcx,
        jitter_dcy=args.jitter_dcy,
        sim_jitter=args.sim_jitter,
        sim_jitter_profile=sim_profile,
        sim_jitter_extrinsics=args.sim_jitter_extrinsics,
        extrinsics_jitter=args.extrinsics_jitter,
        jitter_dx=args.jitter_dx,
        jitter_dy=args.jitter_dy,
        jitter_dz=args.jitter_dz,
        jitter_droll=args.jitter_droll,
        jitter_dpitch=args.jitter_dpitch,
        jitter_dyaw=args.jitter_dyaw,
    )
    sampler = None
    if ddp_enabled:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
            drop_last=False,
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(bool(args.shuffle) if sampler is None else False),
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
        generator=(torch.Generator().manual_seed(int(args.seed)) if args.deterministic and sampler is None else None),
    )

    model_num_queries = model_cfg.num_queries if model_cfg is not None else args.num_queries
    args.num_queries = int(model_num_queries)

    if model_cfg is not None:
        model = build_model(model_cfg)
    else:
        model = RTDETRPose(
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            num_queries=model_num_queries,
            num_decoder_layers=2,
            nhead=4,
            use_uncertainty=bool(args.use_uncertainty),
        )

    if loss_cfg is not None:
        if args.task_aligner and args.task_aligner != "none":
            try:
                loss_cfg.task_aligner = str(args.task_aligner)
            except Exception:
                pass
        losses_fn = build_losses(loss_cfg)
    else:
        losses_fn = Losses(task_aligner=args.task_aligner)

    device_str = str(args.device).strip() if args.device is not None else "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        if is_main:
            print("warning: cuda requested but not available; falling back to cpu")
        device_str = "cpu"
    if ddp_enabled and device_str.startswith("cuda"):
        if torch.cuda.is_available():
            torch.cuda.set_device(int(local_rank))
        device_str = f"cuda:{int(local_rank)}"
    device = torch.device(device_str)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    global_step = 0
    if args.resume_from:
        meta = load_checkpoint_into(model, optim, args.resume_from)
        if meta.get("epoch") is not None:
            try:
                start_epoch = int(meta["epoch"]) + 1
            except Exception:
                start_epoch = 0
        if meta.get("global_step") is not None:
            try:
                global_step = int(meta["global_step"])
            except Exception:
                global_step = 0
        if is_main:
            print(f"resumed_from={meta.get('path')} start_epoch={start_epoch} global_step={global_step}")

    sdft_cfg = None
    teacher_model = None
    if args.self_distill_from:
        # Build a frozen teacher with identical architecture/config.
        if model_cfg is not None:
            teacher_model = build_model(model_cfg)
        else:
            teacher_model = RTDETRPose(
                num_classes=args.num_classes,
                hidden_dim=args.hidden_dim,
                num_queries=model_num_queries,
                num_decoder_layers=2,
                nhead=4,
                use_uncertainty=bool(args.use_uncertainty),
            )
        load_checkpoint_into(teacher_model, None, args.self_distill_from)
        teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)

        keys = tuple(k.strip() for k in str(args.self_distill_keys).split(",") if k.strip())
        sdft_cfg = SdftConfig(
            weight=float(args.self_distill_weight),
            temperature=float(args.self_distill_temperature),
            kl=str(args.self_distill_kl),
            keys=keys,
            logits_weight=float(args.self_distill_logits_weight),
            bbox_weight=float(args.self_distill_bbox_weight),
            other_l1_weight=float(args.self_distill_other_l1_weight),
        )
        if is_main:
            print(
                "self_distill",
                f"from={args.self_distill_from}",
                f"keys={','.join(keys) if keys else '(none)'}",
                f"kl={sdft_cfg.kl}",
                f"temp={sdft_cfg.temperature}",
                f"weight={sdft_cfg.weight}",
            )

    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(local_rank)] if device.type == "cuda" else None,
            output_device=int(local_rank) if device.type == "cuda" else None,
        )

    # AMP setup
    amp_mode = str(args.amp or "none").lower()
    scaler = None
    if amp_mode != "none":
        if device.type != "cuda":
            raise SystemExit("--amp requires --device cuda")
        if amp_mode == "fp16":
            scaler = torch.cuda.amp.GradScaler()
            autocast = torch.cuda.amp.autocast(dtype=torch.float16)
        elif amp_mode == "bf16":
            autocast = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            raise SystemExit(f"unknown --amp mode: {args.amp}")
    else:
        autocast = nullcontext()

    model.train()
    last_loss_dict = None
    last_epoch_avg = None
    last_epoch_steps = 0
    for epoch in range(int(start_epoch), int(args.epochs)):
        if sampler is not None:
            sampler.set_epoch(int(epoch))
        if args.hflip_prob_start is not None and args.hflip_prob_end is not None:
            ds.hflip_prob = compute_linear_schedule(
                float(args.hflip_prob_start),
                float(args.hflip_prob_end),
                int(epoch),
                int(args.epochs),
            )
        running = 0.0
        steps = 0
        max_micro_steps = len(loader)
        if args.max_steps and int(args.max_steps) > 0:
            max_micro_steps = min(max_micro_steps, int(args.max_steps))

        grad_accum = max(1, int(args.grad_accum))

        for images, targets in loader:
            if max_micro_steps and steps >= int(max_micro_steps):
                break

            # Start of accumulation window.
            step_in_window = int(steps) % int(grad_accum)
            window_size = int(min(int(grad_accum), int(max_micro_steps) - int(steps))) if max_micro_steps else int(grad_accum)
            if step_in_window == 0:
                optim.zero_grad(set_to_none=True)

            images = images.to(device)

            sync_step = step_in_window == window_size - 1
            ddp_nosync = ddp_enabled and hasattr(model, "no_sync") and not sync_step
            sync_context = model.no_sync() if ddp_nosync else nullcontext()

            with sync_context:
                with autocast:
                    out = model(images)

                    sdft_total = None
                    sdft_parts = None
                    if teacher_model is not None and sdft_cfg is not None and float(sdft_cfg.weight) != 0.0:
                        with torch.no_grad():
                            with autocast:
                                teacher_out = teacher_model(images)
                        student_out_sdft = out
                        teacher_out_sdft = teacher_out
                        if (
                            "bbox" in sdft_cfg.keys
                            and isinstance(out.get("bbox"), torch.Tensor)
                            and isinstance(teacher_out.get("bbox"), torch.Tensor)
                        ):
                            student_out_sdft = dict(out)
                            teacher_out_sdft = dict(teacher_out)
                            student_out_sdft["bbox"] = out["bbox"].sigmoid()
                            teacher_out_sdft["bbox"] = teacher_out["bbox"].sigmoid()
                        sdft_total, sdft_parts = compute_sdft_loss(student_out_sdft, teacher_out_sdft, sdft_cfg)

                    if args.use_matcher:
                        per_sample = targets.get("per_sample") if isinstance(targets, dict) else targets
                        aligned = build_query_aligned_targets(
                            out["logits"],
                            out["bbox"],
                            per_sample,
                            num_queries=model_num_queries,
                            cost_cls=args.cost_cls,
                            cost_bbox=args.cost_bbox,
                            log_z_pred=out.get("log_z"),
                            rot6d_pred=out.get("rot6d"),
                            cost_z=args.cost_z,
                            cost_rot=args.cost_rot,
                            offsets_pred=out.get("offsets"),
                            k_delta=out.get("k_delta"),
                            cost_t=args.cost_t,
                        )
                        out = dict(out)
                        # For box regression we train in normalized space.
                        out["bbox"] = aligned["bbox_norm"]
                        targets = {
                            "labels": aligned["labels"],
                            "bbox": aligned["bbox"],
                            "mask": aligned["mask"],
                            "z_gt": aligned["z_gt"],
                            "z_mask": aligned["z_mask"],
                            "R_gt": aligned["R_gt"],
                            "rot_mask": aligned["rot_mask"],
                            "offsets": aligned["offsets"],
                            "off_mask": aligned["off_mask"],
                            "t_gt": aligned["t_gt"],
                            "K_gt": aligned["K_gt"],
                            "image_hw": aligned["image_hw"],
                            "K_mask": aligned["K_mask"],
                            "t_mask": aligned["t_mask"],
                            "M_mask": aligned.get("M_mask"),
                            "D_obj_mask": aligned.get("D_obj_mask"),
                        }
                    else:
                        # legacy padded targets
                        targets = {
                            "labels": torch.stack([t["labels"] for t in targets], dim=0).to(device),
                            "bbox": torch.stack([t["bbox"] for t in targets], dim=0).to(device),
                        }

                    loss_dict = losses_fn(out, targets)
                    loss = loss_dict["loss"]
                    if sdft_total is not None and sdft_parts is not None:
                        loss_supervised = loss
                        loss_dict = dict(loss_dict)
                        loss_dict["loss_supervised"] = loss_supervised
                        loss_dict.update(sdft_parts)
                        loss = loss_supervised + float(sdft_cfg.weight) * sdft_total
                        loss_dict["loss"] = loss
                    last_loss_dict = loss_dict

                    if steps == 0 and args.debug_losses and is_main:
                        printable = {
                            k: float(v.detach().cpu())
                            for k, v in loss_dict.items()
                            if hasattr(v, "detach")
                        }
                        print("loss_breakdown", " ".join(f"{k}={v:.6g}" for k, v in sorted(printable.items())))

                    loss_value = float(loss.detach().cpu())
                    running += loss_value
                    loss_for_backward = loss / float(window_size)

                if scaler is not None:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            if sync_step:
                if scaler is not None:
                    scaler.unscale_(optim)
                if args.clip_grad_norm and float(args.clip_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))

                if scaler is not None:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()

                global_step += 1
                if args.lr_warmup_steps and int(args.lr_warmup_steps) > 0:
                    lr_now = compute_warmup_lr(
                        float(args.lr),
                        int(global_step),
                        int(args.lr_warmup_steps),
                        float(args.lr_warmup_init),
                    )
                    for group in optim.param_groups:
                        group["lr"] = lr_now

            steps += 1

            if is_main and (steps == 1 or (args.log_every and steps % int(args.log_every) == 0)):
                avg = running / steps
                print(f"epoch={epoch} step={steps} optim_step={global_step} loss={avg:.4f}")
                if args.metrics_jsonl:
                    losses_out = {k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")} if last_loss_dict is not None else {}
                    report = build_report(
                        losses=losses_out,
                        metrics={"loss_avg": float(avg), "optim_step": int(global_step)},
                        meta={
                            "kind": "train_step",
                            "epoch": int(epoch),
                            "step": int(steps),
                            "optim_step": int(global_step),
                        },
                    )
                    append_jsonl(args.metrics_jsonl, report)

            if (
                is_main
                and sync_step
                and args.checkpoint_bundle_out
                and args.checkpoint_every
                and int(args.checkpoint_every) > 0
            ):
                every = int(args.checkpoint_every)
                if global_step % every == 0:
                    bundle_path = Path(args.checkpoint_bundle_out)
                    stepped = bundle_path.with_name(f"{bundle_path.stem}.step{global_step}{bundle_path.suffix or '.pt'}")
                    save_checkpoint_bundle(
                        stepped,
                        model=unwrap_model(model),
                        optim=optim,
                        args=args,
                        epoch=epoch,
                        global_step=global_step,
                        last_epoch_steps=steps,
                        last_epoch_avg=(running / max(1, steps)),
                        last_loss_dict=last_loss_dict,
                        run_record=run_record,
                    )

        avg = running / max(1, steps)
        last_epoch_avg = float(avg)
        last_epoch_steps = int(steps)
        if is_main:
            print(f"epoch={epoch} done steps={steps} optim_step={global_step} loss={avg:.4f}")
        if is_main and args.metrics_jsonl and last_loss_dict is not None:
            losses_out = {k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")}
            report = build_report(
                losses=losses_out,
                metrics={"loss_avg": float(avg), "steps": int(steps)},
                meta={"kind": "train_epoch", "epoch": int(epoch)},
            )
            append_jsonl(args.metrics_jsonl, report)

    if is_main and (args.metrics_json or args.metrics_csv):
        losses_out = {}
        if last_loss_dict is not None:
            losses_out = {k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")}
        metrics_out = {"epochs": int(args.epochs), "max_steps": int(args.max_steps)}
        if last_epoch_avg is not None:
            metrics_out["loss_avg_last_epoch"] = float(last_epoch_avg)
        summary = build_report(
            losses=losses_out,
            metrics=metrics_out,
            meta={"kind": "train_run", "run_record": run_record},
        )
        if args.metrics_json:
            write_json(args.metrics_json, summary)
        if args.metrics_csv:
            write_csv_row(args.metrics_csv, summary)

    if is_main and args.checkpoint_out:
        ckpt_path = Path(args.checkpoint_out)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(unwrap_model(model).state_dict(), ckpt_path)

    if is_main and args.checkpoint_bundle_out:
        save_checkpoint_bundle(
            args.checkpoint_bundle_out,
            model=unwrap_model(model),
            optim=optim,
            args=args,
            epoch=int(args.epochs) - 1,
            global_step=int(global_step),
            last_epoch_steps=int(last_epoch_steps),
            last_epoch_avg=last_epoch_avg,
            last_loss_dict=last_loss_dict,
            run_record=run_record,
        )

    if is_main and args.onnx_out:
        try:
            from rtdetr_pose.export import export_onnx
        except Exception as exc:  # pragma: no cover
            raise SystemExit("rtdetr_pose.export.export_onnx is required for ONNX export") from exc

        onnx_path = Path(str(args.onnx_out))
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros((1, 3, int(args.image_size), int(args.image_size)), dtype=torch.float32, device=device)
        export_onnx(
            unwrap_model(model).eval(),
            dummy,
            str(onnx_path),
            opset_version=int(args.onnx_opset),
            dynamic_hw=bool(args.onnx_dynamic_hw),
        )
        meta_path = Path(str(args.onnx_meta_out)) if args.onnx_meta_out else onnx_path.with_suffix(onnx_path.suffix + ".meta.json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "timestamp_utc": _now_utc(),
            "onnx": str(onnx_path),
            "opset": int(args.onnx_opset),
            "dynamic_hw": bool(args.onnx_dynamic_hw),
            "dummy_input": {"shape": [1, 3, int(args.image_size), int(args.image_size)], "dtype": "float32"},
            "run_record": run_record,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    if is_main and run_dir is not None:
        (run_dir / "run_record.json").write_text(
            json.dumps(run_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if ddp_enabled:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
