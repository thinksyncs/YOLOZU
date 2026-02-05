import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root.parent))

try:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch is required; install requirements-test.txt") from exc

from rtdetr_pose.dataset import build_manifest
from rtdetr_pose.dataset import extract_full_gt_targets, depth_at_bbox_center
from rtdetr_pose.losses import Losses
from rtdetr_pose.export import export_onnx
from rtdetr_pose.training import build_query_aligned_targets
from rtdetr_pose.model import RTDETRPose

from yolozu.metrics_report import append_jsonl, build_report, write_csv_row, write_json
from yolozu.jitter import default_jitter_profile, sample_intrinsics_jitter, sample_extrinsics_jitter
from yolozu.run_record import build_run_record


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
    parser = argparse.ArgumentParser(description="Minimal RTDETRPose training scaffold (CPU).")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML/JSON config file. Values become argparse defaults; explicit CLI flags override.",
    )
    parser.add_argument("--dataset-root", type=str, default="", help="Path to data/coco128")
    parser.add_argument("--split", type=str, default="train2017")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for training (e.g., cpu, cuda, cuda:0).",
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
    parser.add_argument(
        "--stage-off-steps",
        type=int,
        default=0,
        help="If >0, train offsets-only for this many steps (K loss disabled).",
    )
    parser.add_argument(
        "--stage-k-steps",
        type=int,
        default=0,
        help="If >0, train K-only for this many steps after offsets stage (offset loss disabled).",
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
    parser.add_argument(
        "--mim-mask-prob",
        type=float,
        default=0.0,
        help="Probability of masking each MIM patch (default: 0.0 disables).",
    )
    parser.add_argument(
        "--mim-mask-prob-start",
        type=float,
        default=None,
        help="Optional start value for MIM mask prob schedule.",
    )
    parser.add_argument(
        "--mim-mask-prob-end",
        type=float,
        default=None,
        help="Optional end value for MIM mask prob schedule.",
    )
    parser.add_argument(
        "--mim-mask-size",
        type=int,
        default=16,
        help="Patch size for MIM masking (default: 16).",
    )
    parser.add_argument(
        "--mim-mask-value",
        type=float,
        default=0.0,
        help="Fill value for masked patches (default: 0.0).",
    )
    parser.add_argument(
        "--mim-teacher",
        action="store_true",
        help="Enable self-distillation between masked/unmasked images.",
    )
    parser.add_argument(
        "--mim-loss-weight",
        type=float,
        default=0.0,
        help="Weight for MIM distillation loss (default: 0.0).",
    )
    parser.add_argument(
        "--mim-loss-weight-start",
        type=float,
        default=None,
        help="Optional start value for MIM loss weight schedule.",
    )
    parser.add_argument(
        "--mim-loss-weight-end",
        type=float,
        default=None,
        help="Optional end value for MIM loss weight schedule.",
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
        "--cost-z-start-step",
        type=int,
        default=0,
        help="Delay enabling cost-z until this global step (default: 0).",
    )
    parser.add_argument(
        "--cost-rot",
        type=float,
        default=0.0,
        help="Optional matching cost for rotation (geodesic angle)",
    )
    parser.add_argument(
        "--cost-rot-start-step",
        type=int,
        default=0,
        help="Delay enabling cost-rot until this global step (default: 0).",
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
        help="Allow loading mask/depth arrays from paths (.json/.npy/.png) for z-from-dobj; default keeps lazy paths",
    )
    parser.add_argument(
        "--cost-t",
        type=float,
        default=0.0,
        help="Optional matching cost for translation recovered from (bbox, offsets, z, K')",
    )
    parser.add_argument(
        "--cost-t-start-step",
        type=int,
        default=0,
        help="Delay enabling cost-t until this global step (default: 0).",
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

    # Training-time tricks (optional)
    parser.add_argument(
        "--denoise-queries",
        type=int,
        default=0,
        help="If >0, append denoised copies of GT targets for matching.",
    )
    parser.add_argument(
        "--denoise-bbox-noise",
        type=float,
        default=0.01,
        help="Stddev for bbox noise applied to denoised targets (default: 0.01).",
    )
    parser.add_argument(
        "--denoise-label-noise",
        type=float,
        default=0.0,
        help="Probability to randomize class labels for denoised targets (default: 0.0).",
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
    parser.add_argument(
        "--export-onnx",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Export ONNX after training (default: true).",
    )
    parser.add_argument(
        "--onnx-out",
        default="reports/rtdetr_pose.onnx",
        help="Output path for ONNX export (default: reports/rtdetr_pose.onnx).",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    return parser


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


def compute_mim_schedule(
    *,
    step: int,
    total_steps: int,
    mask_start: float | None,
    mask_end: float | None,
    weight_start: float | None,
    weight_end: float | None,
    default_mask: float,
    default_weight: float,
) -> tuple[float, float]:
    if total_steps <= 0:
        return float(default_mask), float(default_weight)
    mask_val = float(default_mask)
    if mask_start is not None and mask_end is not None:
        mask_val = compute_linear_schedule(float(mask_start), float(mask_end), int(step), int(total_steps))
    weight_val = float(default_weight)
    if weight_start is not None and weight_end is not None:
        weight_val = compute_linear_schedule(float(weight_start), float(weight_end), int(step), int(total_steps))
    return mask_val, weight_val


def compute_stage_weights(
    base: dict[str, float],
    *,
    global_step: int,
    stage_off_steps: int,
    stage_k_steps: int,
) -> tuple[dict[str, float], str]:
    weights = dict(base)
    if stage_off_steps > 0 and global_step < stage_off_steps:
        weights["k"] = 0.0
        return weights, "offsets"
    if stage_k_steps > 0 and global_step < stage_off_steps + stage_k_steps:
        weights["off"] = 0.0
        return weights, "k"
    return weights, "full"


def compute_stage_costs(
    base: dict[str, float],
    *,
    global_step: int,
    cost_z_start_step: int,
    cost_rot_start_step: int,
    cost_t_start_step: int,
) -> dict[str, float]:
    costs = dict(base)
    if global_step < int(cost_z_start_step):
        costs["cost_z"] = 0.0
    if global_step < int(cost_rot_start_step):
        costs["cost_rot"] = 0.0
    if global_step < int(cost_t_start_step):
        costs["cost_t"] = 0.0
    return costs


def apply_denoise_targets(
    targets: list[dict[str, Any]],
    *,
    num_classes: int,
    denoise_count: int,
    bbox_noise: float,
    label_noise: float,
    generator: "torch.Generator | None" = None,
) -> list[dict[str, Any]]:
    if torch is None or not targets or denoise_count <= 0:
        return targets

    out: list[dict[str, Any]] = []
    for tgt in targets:
        if not isinstance(tgt, dict):
            out.append(tgt)
            continue
        gt_labels = tgt.get("gt_labels")
        gt_bbox = tgt.get("gt_bbox")
        if not isinstance(gt_labels, torch.Tensor) or not isinstance(gt_bbox, torch.Tensor):
            out.append(tgt)
            continue
        if gt_labels.numel() == 0 or gt_bbox.numel() == 0:
            out.append(tgt)
            continue

        repeats = int(denoise_count)
        noise = torch.randn(gt_bbox.shape, device=gt_bbox.device, dtype=gt_bbox.dtype) * float(bbox_noise)
        noisy_bbox = torch.clamp(gt_bbox + noise, 0.0, 1.0)
        noisy_bbox = noisy_bbox.repeat((repeats, 1))

        noisy_labels = gt_labels.repeat(repeats)
        if label_noise and float(label_noise) > 0:
            rand_mask = torch.rand(noisy_labels.shape, device=noisy_labels.device, dtype=torch.float32) < float(label_noise)
            if bool(rand_mask.any()):
                noisy_labels[rand_mask] = torch.randint(
                    low=0,
                    high=int(num_classes),
                    size=(int(rand_mask.sum().item()),),
                    device=noisy_labels.device,
                    dtype=noisy_labels.dtype,
                )

        new_tgt = dict(tgt)
        new_tgt["gt_labels"] = torch.cat([gt_labels, noisy_labels], dim=0)
        new_tgt["gt_bbox"] = torch.cat([gt_bbox, noisy_bbox], dim=0)

        for key in (
            "gt_z",
            "gt_z_mask",
            "gt_R",
            "gt_R_mask",
            "gt_t",
            "gt_t_mask",
            "gt_offsets",
            "gt_offsets_mask",
            "gt_M_mask",
            "gt_D_obj_mask",
        ):
            value = tgt.get(key)
            if isinstance(value, torch.Tensor) and value.shape[0] == gt_labels.shape[0]:
                rep_shape = (repeats,) + (1,) * (value.ndim - 1)
                value_rep = value.repeat(rep_shape)
                new_tgt[key] = torch.cat([value, value_rep], dim=0)

        out.append(new_tgt)
    return out


def export_onnx_after_training(model: "torch.nn.Module", *, image_size: int, output_path: str | Path, opset_version: int) -> Path:
    device = next(model.parameters()).device
    dummy = torch.zeros((1, 3, int(image_size), int(image_size)), dtype=torch.float32, device=device)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, dummy, output_path, opset_version=int(opset_version))
    return output_path


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
        mim_mask_prob=0.0,
        mim_mask_size=16,
        mim_mask_value=0.0,
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
        self.mim_mask_prob = float(mim_mask_prob)
        self.mim_mask_size = int(mim_mask_size)
        self.mim_mask_value = float(mim_mask_value)

    def _apply_mim_mask(self, image: "torch.Tensor", *, generator: "torch.Generator") -> tuple["torch.Tensor", float]:
        if self.mim_mask_prob <= 0.0 or self.mim_mask_size <= 0:
            return image, 0.0

        _, h, w = image.shape
        grid_h = max(1, int(math.ceil(float(h) / float(self.mim_mask_size))))
        grid_w = max(1, int(math.ceil(float(w) / float(self.mim_mask_size))))
        mask = torch.rand((grid_h, grid_w), generator=generator) < float(self.mim_mask_prob)
        if not bool(mask.any()):
            return image, 0.0

        mask_full = mask.repeat_interleave(self.mim_mask_size, dim=0).repeat_interleave(self.mim_mask_size, dim=1)
        mask_full = mask_full[:h, :w]
        masked = image.clone()
        masked[:, mask_full] = float(self.mim_mask_value)
        ratio = float(mask_full.float().mean().item())
        return masked, ratio

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
        if isinstance(value, (str, Path)):
            path = Path(value)
            suffix = path.suffix.lower()
            if suffix == ".json":
                import json

                try:
                    return json.loads(path.read_text())
                except Exception:
                    return None
            if suffix == ".png":
                try:
                    from PIL import Image
                except Exception as exc:
                    raise SystemExit(
                        "Pillow is required for PNG masks. Install it (e.g. pip install Pillow) or use .npy/.json."
                    ) from exc
                try:
                    import numpy as np
                except Exception:
                    np = None  # type: ignore
                try:
                    img = Image.open(path).convert("L")
                except Exception:
                    return None
                if np is not None:
                    return np.asarray(img)
                width, height = img.size
                data = list(img.getdata())
                return [data[i * width : (i + 1) * width] for i in range(height)]
            if suffix in (".npy", ".npz"):
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

        image_raw = image.clone()
        image, mim_mask_ratio = self._apply_mim_mask(image, generator=gen)

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
                "image_raw": image_raw,
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
                "mim_mask_ratio": mim_mask_ratio,
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

        return {
            "image": image,
            "image_raw": image_raw,
            "targets": {"labels": labels, "bbox": bbox},
            "mim_mask_ratio": mim_mask_ratio,
        }


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
    image_raw = None
    if all("image_raw" in item and isinstance(item["image_raw"], torch.Tensor) for item in batch):
        image_raw = torch.stack([item["image_raw"] for item in batch], dim=0)
    mim_mask_ratio = torch.tensor(
        [float(item.get("mim_mask_ratio", 0.0)) for item in batch],
        dtype=torch.float32,
    )
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
        out = {"per_sample": targets, "padded": padded, "mim_mask_ratio": mim_mask_ratio}
        if image_raw is not None:
            out["image_raw"] = image_raw
        return images, out

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

    out = {"per_sample": targets, "padded": padded, "mim_mask_ratio": mim_mask_ratio}
    if image_raw is not None:
        out["image_raw"] = image_raw
    return images, out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

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
    )

    torch.manual_seed(args.seed)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = repo_root / "data" / "coco128"
        if not dataset_root.exists():
            dataset_root = repo_root.parent / "data" / "coco128"

    model_cfg = None
    if args.config:
        try:
            from rtdetr_pose.config import load_config
        except Exception:
            load_config = None
        if load_config is not None:
            try:
                model_cfg = load_config(args.config).model
            except Exception:
                model_cfg = None
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
        mim_mask_prob=args.mim_mask_prob,
        mim_mask_size=args.mim_mask_size,
        mim_mask_value=args.mim_mask_value,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=bool(args.shuffle),
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
        generator=(torch.Generator().manual_seed(int(args.seed)) if args.deterministic else None),
    )

    model_num_queries = model_cfg.num_queries if model_cfg is not None else args.num_queries
    args.num_queries = int(model_num_queries)

    model = RTDETRPose(
        num_classes=(model_cfg.num_classes if model_cfg is not None else args.num_classes),
        hidden_dim=(model_cfg.hidden_dim if model_cfg is not None else args.hidden_dim),
        num_queries=model_num_queries,
        num_decoder_layers=(model_cfg.num_decoder_layers if model_cfg is not None else 2),
        nhead=(model_cfg.nhead if model_cfg is not None else 4),
        use_uncertainty=(model_cfg.use_uncertainty if model_cfg is not None else bool(args.use_uncertainty)),
        stem_channels=(getattr(model_cfg, "stem_channels", None) if model_cfg is not None else None) or 32,
        backbone_channels=(
            tuple(getattr(model_cfg, "backbone_channels", ()))
            if model_cfg is not None and getattr(model_cfg, "backbone_channels", None) is not None
            else (64, 128, 256)
        ),
        stage_blocks=(
            tuple(getattr(model_cfg, "stage_blocks", ()))
            if model_cfg is not None and getattr(model_cfg, "stage_blocks", None) is not None
            else (1, 2, 2)
        ),
        use_sppf=(getattr(model_cfg, "use_sppf", True) if model_cfg is not None else True),
        num_encoder_layers=(
            getattr(model_cfg, "num_encoder_layers", None) if model_cfg is not None else None
        )
        or 1,
        encoder_dim_feedforward=(
            getattr(model_cfg, "encoder_dim_feedforward", None) if model_cfg is not None else None
        ),
        decoder_dim_feedforward=(
            getattr(model_cfg, "decoder_dim_feedforward", None) if model_cfg is not None else None
        ),
        use_level_embed=(getattr(model_cfg, "use_level_embed", True) if model_cfg is not None else True),
    )
    losses_fn = Losses(task_aligner=args.task_aligner)
    base_weights = dict(losses_fn.weights)

    device_str = str(args.device).strip() if args.device is not None else "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("warning: cuda requested but not available; falling back to cpu")
        device_str = "cpu"
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
        print(f"resumed_from={meta.get('path')} start_epoch={start_epoch} global_step={global_step}")

    model.train()
    last_loss_dict = None
    last_epoch_avg = None
    last_stage = None
    total_steps_est = 0
    if args.max_steps and int(args.max_steps) > 0:
        total_steps_est = int(args.max_steps) * int(args.epochs)
    else:
        total_steps_est = len(loader) * int(args.epochs)
    for epoch in range(int(start_epoch), int(args.epochs)):
        if args.hflip_prob_start is not None and args.hflip_prob_end is not None:
            ds.hflip_prob = compute_linear_schedule(
                float(args.hflip_prob_start),
                float(args.hflip_prob_end),
                int(epoch),
                int(args.epochs),
            )
        running = 0.0
        steps = 0
        for images, targets in loader:
            mim_mask_prob, mim_weight = compute_mim_schedule(
                step=int(global_step),
                total_steps=int(total_steps_est),
                mask_start=args.mim_mask_prob_start,
                mask_end=args.mim_mask_prob_end,
                weight_start=args.mim_loss_weight_start,
                weight_end=args.mim_loss_weight_end,
                default_mask=args.mim_mask_prob,
                default_weight=args.mim_loss_weight,
            )
            ds.mim_mask_prob = float(mim_mask_prob)
            images = images.to(device)
            mim_ratio = None
            if isinstance(targets, dict) and "mim_mask_ratio" in targets:
                try:
                    mim_ratio = float(targets["mim_mask_ratio"].mean().detach().cpu())
                except Exception:
                    mim_ratio = None
            out = model(images)
            mim_loss = None
            if args.mim_teacher and float(mim_weight) > 0 and isinstance(targets, dict):
                image_raw = targets.get("image_raw")
                if isinstance(image_raw, torch.Tensor):
                    ratio = targets.get("mim_mask_ratio")
                    if ratio is None or bool((ratio > 0).any()):
                        was_training = model.training
                        if was_training:
                            model.eval()
                        with torch.no_grad():
                            teacher_out = model(image_raw.to(device))
                        if was_training:
                            model.train()
                        loss_items = []
                        if "logits" in out and "logits" in teacher_out:
                            loss_items.append(F.mse_loss(out["logits"], teacher_out["logits"]))
                        if "bbox" in out and "bbox" in teacher_out:
                            loss_items.append(F.l1_loss(out["bbox"], teacher_out["bbox"]))
                        if loss_items:
                            mim_loss = sum(loss_items)

            if args.use_matcher:
                staged_costs = compute_stage_costs(
                    {
                        "cost_z": float(args.cost_z),
                        "cost_rot": float(args.cost_rot),
                        "cost_t": float(args.cost_t),
                    },
                    global_step=int(global_step),
                    cost_z_start_step=int(args.cost_z_start_step),
                    cost_rot_start_step=int(args.cost_rot_start_step),
                    cost_t_start_step=int(args.cost_t_start_step),
                )
                per_sample = targets.get("per_sample") if isinstance(targets, dict) else targets
                if (
                    args.denoise_queries
                    and isinstance(per_sample, list)
                    and int(args.denoise_queries) > 0
                ):
                    per_sample = apply_denoise_targets(
                        per_sample,
                        num_classes=int(args.num_classes),
                        denoise_count=int(args.denoise_queries),
                        bbox_noise=float(args.denoise_bbox_noise),
                        label_noise=float(args.denoise_label_noise),
                    )
                aligned = build_query_aligned_targets(
                    out["logits"],
                    out["bbox"],
                    per_sample,
                    num_queries=model_num_queries,
                    cost_cls=args.cost_cls,
                    cost_bbox=args.cost_bbox,
                    log_z_pred=out.get("log_z"),
                    rot6d_pred=out.get("rot6d"),
                    cost_z=staged_costs["cost_z"],
                    cost_rot=staged_costs["cost_rot"],
                    offsets_pred=out.get("offsets"),
                    k_delta=out.get("k_delta"),
                    cost_t=staged_costs["cost_t"],
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

            staged_weights, stage = compute_stage_weights(
                base_weights,
                global_step=int(global_step),
                stage_off_steps=int(args.stage_off_steps),
                stage_k_steps=int(args.stage_k_steps),
            )
            losses_fn.weights = staged_weights
            if stage != last_stage:
                print(f"stage={stage}")
                last_stage = stage

            loss_dict = losses_fn(out, targets)
            loss = loss_dict["loss"]
            if mim_loss is not None:
                loss = loss + float(mim_weight) * mim_loss
                loss_dict = dict(loss_dict)
                loss_dict["loss_mim"] = mim_loss
                loss_dict["loss"] = loss
            last_loss_dict = loss_dict

            if steps == 0 and args.debug_losses:
                printable = {
                    k: float(v.detach().cpu())
                    for k, v in loss_dict.items()
                    if hasattr(v, "detach")
                }
                print("loss_breakdown", " ".join(f"{k}={v:.6g}" for k, v in sorted(printable.items())))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad_norm and float(args.clip_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
            optim.step()

            if args.lr_warmup_steps and int(args.lr_warmup_steps) > 0:
                lr_now = compute_warmup_lr(
                    float(args.lr),
                    int(global_step),
                    int(args.lr_warmup_steps),
                    float(args.lr_warmup_init),
                )
                for group in optim.param_groups:
                    group["lr"] = lr_now

            running += float(loss.detach().cpu())
            steps += 1
            global_step += 1

            if steps == 1 or (args.log_every and steps % int(args.log_every) == 0):
                avg = running / steps
                suffix = ""
                if mim_ratio is not None:
                    suffix = f" mim_mask_ratio={mim_ratio:.3f}"
                suffix += f" mim_mask_prob={mim_mask_prob:.3f} mim_weight={mim_weight:.3f}"
                print(f"epoch={epoch} step={steps} global_step={global_step} loss={avg:.4f}{suffix}")
                if args.metrics_jsonl:
                    losses_out = {k: float(v.detach().cpu()) for k, v in loss_dict.items() if hasattr(v, "detach")}
                    metrics_out = {"loss_avg": float(avg)}
                    if mim_ratio is not None:
                        metrics_out["mim_mask_ratio"] = float(mim_ratio)
                    metrics_out["mim_mask_prob"] = float(mim_mask_prob)
                    metrics_out["mim_weight"] = float(mim_weight)
                    report = build_report(
                        losses=losses_out,
                        metrics=metrics_out,
                        meta={
                            "kind": "train_step",
                            "epoch": int(epoch),
                            "step": int(steps),
                            "global_step": int(global_step),
                            "stage": stage,
                        },
                    )
                    append_jsonl(args.metrics_jsonl, report)

            if args.checkpoint_bundle_out and args.checkpoint_every and int(args.checkpoint_every) > 0:
                every = int(args.checkpoint_every)
                if global_step % every == 0:
                    bundle_path = Path(args.checkpoint_bundle_out)
                    stepped = bundle_path.with_name(f"{bundle_path.stem}.step{global_step}{bundle_path.suffix or '.pt'}")
                    save_checkpoint_bundle(
                        stepped,
                        model=model,
                        optim=optim,
                        args=args,
                        epoch=epoch,
                        global_step=global_step,
                        last_epoch_steps=steps,
                        last_epoch_avg=(running / max(1, steps)),
                        last_loss_dict=loss_dict,
                        run_record=run_record,
                    )

            if args.max_steps and steps >= int(args.max_steps):
                break

        avg = running / max(1, steps)
        last_epoch_avg = float(avg)
        print(f"epoch={epoch} done steps={steps} loss={avg:.4f}")
        if args.metrics_jsonl and last_loss_dict is not None:
            losses_out = {k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")}
            metrics_out = {"loss_avg": float(avg), "steps": int(steps)}
            if mim_ratio is not None:
                metrics_out["mim_mask_ratio"] = float(mim_ratio)
            metrics_out["mim_mask_prob"] = float(mim_mask_prob)
            metrics_out["mim_weight"] = float(mim_weight)
            report = build_report(
                losses=losses_out,
                metrics=metrics_out,
                meta={"kind": "train_epoch", "epoch": int(epoch)},
            )
            append_jsonl(args.metrics_jsonl, report)

    if args.metrics_json or args.metrics_csv:
        losses_out = {}
        if last_loss_dict is not None:
            losses_out = {k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")}
        metrics_out = {
            "epochs": int(args.epochs),
            "max_steps": int(args.max_steps),
            "stage_off_steps": int(args.stage_off_steps),
            "stage_k_steps": int(args.stage_k_steps),
            "mim_mask_prob_start": args.mim_mask_prob_start,
            "mim_mask_prob_end": args.mim_mask_prob_end,
            "mim_loss_weight_start": args.mim_loss_weight_start,
            "mim_loss_weight_end": args.mim_loss_weight_end,
        }
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

    if args.checkpoint_out:
        ckpt_path = Path(args.checkpoint_out)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

    if args.checkpoint_bundle_out:
        save_checkpoint_bundle(
            args.checkpoint_bundle_out,
            model=model,
            optim=optim,
            args=args,
            epoch=int(args.epochs) - 1,
            global_step=int(global_step),
            last_epoch_steps=int(args.max_steps),
            last_epoch_avg=last_epoch_avg,
            last_loss_dict=last_loss_dict,
            run_record=run_record,
        )

    if args.export_onnx:
        onnx_path = export_onnx_after_training(
            model,
            image_size=int(args.image_size),
            output_path=args.onnx_out,
            opset_version=int(args.onnx_opset),
        )
        print(onnx_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
