from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .config import TTTConfig
from .tent import TentConfig, TentRunner

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class TTTReport:
    enabled: bool
    method: str
    reset: str
    steps_requested: int
    steps_run: int
    batches_used: int
    seconds: float
    losses: list[float]
    mask_ratio: float | None
    updated_param_count: int | None
    warnings: list[str]
    stopped_early: bool = False
    stop_reason: str | None = None
    step_metrics: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        losses = list(self.losses or [])
        if losses:
            data["loss_summary"] = {
                "count": int(len(losses)),
                "min": float(min(losses)),
                "mean": float(sum(losses) / float(len(losses))),
                "max": float(max(losses)),
                "last": float(losses[-1]),
            }
        else:
            data["loss_summary"] = {"count": 0, "min": None, "mean": None, "max": None, "last": None}

        max_keep = 50
        if len(losses) > max_keep:
            data["losses_total"] = int(len(losses))
            data["losses_truncated"] = True
            data["losses"] = losses[:max_keep]
        else:
            data["losses_total"] = int(len(losses))
            data["losses_truncated"] = False

        step_metrics = list(self.step_metrics or [])
        if len(step_metrics) > max_keep:
            data["step_metrics_total"] = int(len(step_metrics))
            data["step_metrics_truncated"] = True
            data["step_metrics"] = step_metrics[:max_keep]
        else:
            data["step_metrics_total"] = int(len(step_metrics))
            data["step_metrics_truncated"] = False
        return data


def _ensure_torch():
    if torch is None:
        raise RuntimeError("torch is required for TTT")


def _count_unique_params(params: Iterable["torch.Tensor"]) -> int:
    seen: set[int] = set()
    total = 0
    for param in params:
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        total += int(param.numel())
    return total


def _take_batches(loader: Iterable[Any], *, max_batches: int) -> list[Any]:
    if max_batches <= 0:
        raise ValueError("max_batches must be > 0")
    batches = []
    it = iter(loader)
    for _ in range(int(max_batches)):
        try:
            batches.append(next(it))
        except StopIteration:
            break
    return batches


def _snapshot_params(params: Iterable["torch.Tensor"]) -> list[tuple["torch.Tensor", "torch.Tensor"]]:
    _ensure_torch()
    snap: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for p in params:
            snap.append((p, p.detach().clone()))
    return snap


def _restore_params(snapshot: Iterable[tuple["torch.Tensor", "torch.Tensor"]]) -> None:
    _ensure_torch()
    with torch.no_grad():
        for p, value in snapshot:
            p.copy_(value)


def _snapshot_norm_buffers(model: Any) -> list[tuple["torch.Tensor", "torch.Tensor"]]:
    _ensure_torch()
    snap: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        named_buffers = model.named_buffers()
    except Exception:
        return snap

    with torch.no_grad():
        for name, buffer in named_buffers:
            if buffer is None:
                continue
            name = str(name)
            if not name.endswith(("running_mean", "running_var", "num_batches_tracked")):
                continue
            try:
                snap.append((buffer, buffer.detach().clone()))
            except Exception:  # pragma: no cover
                continue
    return snap


def _restore_buffers(snapshot: Iterable[tuple["torch.Tensor", "torch.Tensor"]]) -> None:
    _ensure_torch()
    with torch.no_grad():
        for buffer, value in snapshot:
            buffer.copy_(value)


def _global_l2_update_norm(snapshot: Iterable[tuple["torch.Tensor", "torch.Tensor"]]) -> float:
    _ensure_torch()
    total = None
    for p, base in snapshot:
        delta = (p.detach() - base).to(dtype=torch.float32)
        if total is None:
            total = torch.zeros((), device=delta.device, dtype=torch.float32)
        total = total + (delta * delta).sum()
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach().cpu().item())


def run_ttt(adapter: Any, records: list[dict[str, Any]], *, config: TTTConfig) -> TTTReport:
    if not config.enabled:
        return TTTReport(
            enabled=False,
            method=str(config.method),
            reset=str(config.reset),
            steps_requested=int(config.steps),
            steps_run=0,
            batches_used=0,
            seconds=0.0,
            losses=[],
            mask_ratio=None,
            updated_param_count=None,
            warnings=[],
            stopped_early=False,
            stop_reason=None,
            step_metrics=[],
        )

    _ensure_torch()

    if not hasattr(adapter, "get_model") or not hasattr(adapter, "build_loader"):
        raise RuntimeError("TTT requires an adapter with get_model() and build_loader()")

    model = adapter.get_model()
    if model is None:
        raise RuntimeError("TTT requires a non-null model from adapter.get_model()")

    loader = adapter.build_loader(records, batch_size=int(config.batch_size))
    batches = _take_batches(loader, max_batches=int(config.max_batches))
    if not batches:
        raise RuntimeError("TTT requires at least one batch from adapter.build_loader()")

    method = (config.method or "tent").lower()
    if method not in ("tent", "mim"):
        raise ValueError("TTT method must be one of: tent, mim")

    steps = int(config.steps)
    if steps <= 0:
        raise ValueError("steps must be > 0")

    warnings: list[str] = []
    losses: list[float] = []
    mask_ratios: list[float] = []
    updated_param_count: int | None = None
    step_metrics: list[dict[str, Any]] = []
    stopped_early = False
    stop_reason: str | None = None

    was_training = bool(getattr(model, "training", False))

    generator = None
    if config.seed is not None:
        try:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(config.seed))
        except Exception:  # pragma: no cover
            warnings.append("failed_to_seed_generator")
            generator = None

    start = time.time()
    try:
        initial_loss: float | None = None
        if method == "tent":
            runner = TentRunner(
                model,
                config=TentConfig(
                    lr=float(config.lr),
                    include=config.include,
                    exclude=config.exclude,
                    update_filter=str(config.update_filter),
                    max_grad_norm=config.max_grad_norm,
                ),
            )
            params = list(getattr(runner, "params", []))
            if not params:
                raise RuntimeError("no parameters selected for TTT")
            updated_param_count = int(getattr(runner, "updated_param_count", _count_unique_params(params)))
            base_snapshot = _snapshot_params(params)
            for step_idx in range(steps):
                batch = batches[step_idx % len(batches)]
                pre_snapshot = _snapshot_params(params)
                pre_buffer_snapshot = _snapshot_norm_buffers(model) if bool(config.rollback_on_stop) else []
                metrics = runner.adapt_step(batch)
                loss_value = float(metrics.get("loss_entropy", 0.0))
                if initial_loss is None:
                    initial_loss = loss_value

                step_grad_norm = metrics.get("grad_norm")
                step_grad_norm_clipped = metrics.get("grad_norm_clipped")
                update_norm = _global_l2_update_norm(pre_snapshot)
                total_update_norm = _global_l2_update_norm(base_snapshot)
                step_entry: dict[str, Any] = {
                    "step": int(step_idx),
                    "loss": float(loss_value),
                    "grad_norm": (float(step_grad_norm) if step_grad_norm is not None else None),
                    "grad_norm_clipped": (float(step_grad_norm_clipped) if step_grad_norm_clipped is not None else None),
                    "update_norm": float(update_norm),
                    "total_update_norm": float(total_update_norm),
                    "rolled_back": False,
                }

                non_finite_fields: list[str] = []
                if bool(config.stop_on_non_finite):
                    for key, value in (
                        ("loss", loss_value),
                        ("grad_norm", step_entry["grad_norm"]),
                        ("grad_norm_clipped", step_entry["grad_norm_clipped"]),
                        ("update_norm", update_norm),
                        ("total_update_norm", total_update_norm),
                    ):
                        if value is None:
                            continue
                        if not math.isfinite(float(value)):
                            non_finite_fields.append(str(key))
                if non_finite_fields:
                    stopped_early = True
                    stop_reason = "non_finite_metrics"
                elif config.max_update_norm is not None and update_norm > float(config.max_update_norm):
                    stopped_early = True
                    stop_reason = "max_update_norm_exceeded"
                elif config.max_total_update_norm is not None and total_update_norm > float(config.max_total_update_norm):
                    stopped_early = True
                    stop_reason = "max_total_update_norm_exceeded"
                elif (
                    config.max_loss_ratio is not None
                    and initial_loss is not None
                    and initial_loss > 0.0
                    and loss_value > (initial_loss * float(config.max_loss_ratio))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_ratio_exceeded"
                elif (
                    config.max_loss_increase is not None
                    and initial_loss is not None
                    and loss_value > (initial_loss + float(config.max_loss_increase))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_increase_exceeded"

                if stopped_early:
                    step_entry["rolled_back"] = bool(config.rollback_on_stop)
                    step_entry["stop_reason"] = str(stop_reason)
                    step_entry["non_finite_fields"] = non_finite_fields or None
                    warnings.append(str(stop_reason))
                    if non_finite_fields:
                        warnings.append(f"non_finite_fields:{','.join(non_finite_fields)}")
                    if bool(config.rollback_on_stop):
                        _restore_params(pre_snapshot)
                        _restore_buffers(pre_buffer_snapshot)
                    step_metrics.append(step_entry)
                    break

                losses.append(float(loss_value))
                step_metrics.append(step_entry)
        else:
            from .ttt_mim import select_parameters, ttt_mim_step

            params = select_parameters(
                model,
                update_filter=str(config.update_filter),
                include=config.include,
                exclude=config.exclude,
            )
            if not params:
                raise RuntimeError("no parameters selected for TTT")
            optimizer = torch.optim.Adam(params, lr=float(config.lr))
            updated_param_count = _count_unique_params(params)
            base_snapshot = _snapshot_params(params)
            for step_idx in range(steps):
                x = batches[step_idx % len(batches)]
                pre_snapshot = _snapshot_params(params)
                pre_buffer_snapshot = _snapshot_norm_buffers(model) if bool(config.rollback_on_stop) else []
                loss, _mask_ratio, extra = ttt_mim_step(
                    model,
                    optimizer,
                    x,
                    mask_prob=float(config.mim_mask_prob),
                    patch_size=int(config.mim_patch_size),
                    mask_value=float(config.mim_mask_value),
                    generator=generator,
                    max_grad_norm=config.max_grad_norm,
                )
                loss_value = float(loss.detach().cpu().item())
                if initial_loss is None:
                    initial_loss = loss_value

                update_norm = _global_l2_update_norm(pre_snapshot)
                total_update_norm = _global_l2_update_norm(base_snapshot)
                step_grad_norm = extra.get("grad_norm") if isinstance(extra, dict) else None
                step_grad_norm_clipped = extra.get("grad_norm_clipped") if isinstance(extra, dict) else None
                step_entry = {
                    "step": int(step_idx),
                    "loss": float(loss_value),
                    "mask_ratio": float(_mask_ratio),
                    "grad_norm": (float(step_grad_norm) if step_grad_norm is not None else None),
                    "grad_norm_clipped": (float(step_grad_norm_clipped) if step_grad_norm_clipped is not None else None),
                    "update_norm": float(update_norm),
                    "total_update_norm": float(total_update_norm),
                    "rolled_back": False,
                }

                non_finite_fields = []
                if bool(config.stop_on_non_finite):
                    for key, value in (
                        ("loss", loss_value),
                        ("grad_norm", step_entry["grad_norm"]),
                        ("grad_norm_clipped", step_entry["grad_norm_clipped"]),
                        ("update_norm", update_norm),
                        ("total_update_norm", total_update_norm),
                    ):
                        if value is None:
                            continue
                        if not math.isfinite(float(value)):
                            non_finite_fields.append(str(key))
                if non_finite_fields:
                    stopped_early = True
                    stop_reason = "non_finite_metrics"
                elif config.max_update_norm is not None and update_norm > float(config.max_update_norm):
                    stopped_early = True
                    stop_reason = "max_update_norm_exceeded"
                elif config.max_total_update_norm is not None and total_update_norm > float(config.max_total_update_norm):
                    stopped_early = True
                    stop_reason = "max_total_update_norm_exceeded"
                elif (
                    config.max_loss_ratio is not None
                    and initial_loss is not None
                    and initial_loss > 0.0
                    and loss_value > (initial_loss * float(config.max_loss_ratio))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_ratio_exceeded"
                elif (
                    config.max_loss_increase is not None
                    and initial_loss is not None
                    and loss_value > (initial_loss + float(config.max_loss_increase))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_increase_exceeded"

                if stopped_early:
                    step_entry["rolled_back"] = bool(config.rollback_on_stop)
                    step_entry["stop_reason"] = str(stop_reason)
                    step_entry["non_finite_fields"] = non_finite_fields or None
                    warnings.append(str(stop_reason))
                    if non_finite_fields:
                        warnings.append(f"non_finite_fields:{','.join(non_finite_fields)}")
                    if bool(config.rollback_on_stop):
                        _restore_params(pre_snapshot)
                        _restore_buffers(pre_buffer_snapshot)
                    step_metrics.append(step_entry)
                    break

                losses.append(float(loss_value))
                mask_ratios.append(float(_mask_ratio))
                step_metrics.append(step_entry)
    finally:
        if not was_training:
            try:
                model.eval()
            except Exception:  # pragma: no cover
                warnings.append("failed_to_restore_eval")

    seconds = float(time.time() - start)

    mask_ratio = None
    if mask_ratios:
        mask_ratio = float(sum(mask_ratios) / float(len(mask_ratios)))

    return TTTReport(
        enabled=True,
        method=method,
        reset=str(config.reset),
        steps_requested=steps,
        steps_run=len(losses),
        batches_used=len(batches),
        seconds=seconds,
        losses=losses,
        mask_ratio=mask_ratio,
        updated_param_count=updated_param_count,
        warnings=warnings,
        stopped_early=bool(stopped_early),
        stop_reason=stop_reason,
        step_metrics=step_metrics,
    )
