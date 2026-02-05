from __future__ import annotations

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
    steps_requested: int
    steps_run: int
    batches_used: int
    seconds: float
    losses: list[float]
    mask_ratio: float | None
    updated_param_count: int | None
    warnings: list[str]

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


def run_ttt(adapter: Any, records: list[dict[str, Any]], *, config: TTTConfig) -> TTTReport:
    if not config.enabled:
        return TTTReport(
            enabled=False,
            method=str(config.method),
            steps_requested=int(config.steps),
            steps_run=0,
            batches_used=0,
            seconds=0.0,
            losses=[],
            mask_ratio=None,
            updated_param_count=None,
            warnings=[],
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
        if method == "tent":
            runner = TentRunner(
                model,
                config=TentConfig(
                    lr=float(config.lr),
                    include=config.include,
                    exclude=config.exclude,
                    update_filter=str(config.update_filter),
                ),
            )
            for step_idx in range(steps):
                batch = batches[step_idx % len(batches)]
                metrics = runner.adapt_step(batch)
                losses.append(float(metrics.get("loss_entropy", 0.0)))
            log = runner.maybe_log() or {}
            if "updated_param_count" in log:
                updated_param_count = int(log["updated_param_count"])
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
            for step_idx in range(steps):
                x = batches[step_idx % len(batches)]
                loss, _mask_ratio = ttt_mim_step(
                    model,
                    optimizer,
                    x,
                    mask_prob=float(config.mim_mask_prob),
                    patch_size=int(config.mim_patch_size),
                    mask_value=float(config.mim_mask_value),
                    generator=generator,
                )
                losses.append(float(loss.detach().cpu().item()))
                mask_ratios.append(float(_mask_ratio))
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
        steps_requested=steps,
        steps_run=len(losses),
        batches_used=len(batches),
        seconds=seconds,
        losses=losses,
        mask_ratio=mask_ratio,
        updated_param_count=updated_param_count,
        warnings=warnings,
    )
