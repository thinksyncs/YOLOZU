from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _utc_run_id() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("demo continual requires torch (try: pip install 'yolozu[demo]' or pip install torch)") from exc
    return torch


@dataclass(frozen=True)
class ContinualDemoReport:
    schema_version: int
    settings: dict[str, Any]
    metrics: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"kind": "continual_demo", "schema_version": int(self.schema_version), "settings": self.settings, "metrics": self.metrics}


def run_continual_demo(
    *,
    output: str | Path | None = None,
    seed: int = 0,
    device: str = "cpu",
    method: str = "ewc_replay",
    steps_a: int = 200,
    steps_b: int = 200,
    batch_size: int = 64,
    hidden: int = 32,
    lr: float = 1e-2,
    corr: float = 2.0,
    noise: float = 0.6,
    n_train: int = 4096,
    n_eval: int = 1024,
    ewc_lambda: float = 20.0,
    fisher_batches: int = 64,
    replay_capacity: int = 512,
    replay_k: int = 64,
) -> Path | None:
    """Toy continual-learning demo on a synthetic domain shift (CPU-friendly).

    Two domains have a spurious correlation that flips between A and B. Sequential fine-tuning
    tends to forget A; EWC and/or replay mitigate that forgetting.

    This demo requires torch (CPU is fine).
    """

    torch = _require_torch()
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore

    from yolozu.continual_regularizers import EwcAccumulator, ewc_penalty
    from yolozu.replay_buffer import ReplayBuffer

    device_t = torch.device(str(device))

    if method not in {"naive", "ewc", "replay", "ewc_replay"}:
        raise ValueError("method must be one of: naive, ewc, replay, ewc_replay")

    # Seeds
    torch.manual_seed(int(seed))

    def _make_domain(*, flip_spurious: bool, n: int) -> tuple[Any, Any]:
        # Binary labels.
        y = torch.randint(low=0, high=2, size=(int(n),), device=device_t)
        y_sign = (y.float() * 2.0 - 1.0)  # {-1, +1}
        # True signal: x aligns with label.
        x = y_sign + float(noise) * torch.randn(int(n), device=device_t)
        # Spurious feature: y aligns with label but flips across domains.
        s = y_sign * ( -1.0 if bool(flip_spurious) else 1.0)
        z = float(corr) * s + float(noise) * torch.randn(int(n), device=device_t)
        X = torch.stack([x, z], dim=1)
        return X, y

    X_a_train, y_a_train = _make_domain(flip_spurious=False, n=int(n_train))
    X_b_train, y_b_train = _make_domain(flip_spurious=True, n=int(n_train))
    X_a_eval, y_a_eval = _make_domain(flip_spurious=False, n=int(n_eval))
    X_b_eval, y_b_eval = _make_domain(flip_spurious=True, n=int(n_eval))

    model = nn.Sequential(nn.Linear(2, int(hidden)), nn.ReLU(), nn.Linear(int(hidden), 2)).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    def _acc(model_: Any, X: Any, y: Any) -> float:
        model_.eval()
        with torch.no_grad():
            logits = model_(X)
            pred = logits.argmax(dim=1)
            return float((pred == y).float().mean().item())

    def _train_steps(
        *,
        X: Any,
        y: Any,
        steps: int,
        ewc_state: Any | None,
        replay: ReplayBuffer | None,
    ) -> None:
        model.train()
        n = int(X.shape[0])
        for _ in range(int(steps)):
            idx = torch.randint(low=0, high=n, size=(int(batch_size),), device=device_t)
            xb = X.index_select(0, idx)
            yb = y.index_select(0, idx)

            if replay is not None and replay_k > 0 and len(replay) > 0:
                items = replay.sample(int(replay_k))
                if items:
                    xr = torch.stack([torch.tensor(it["x"], device=device_t) for it in items], dim=0)
                    yr = torch.tensor([int(it["y"]) for it in items], device=device_t, dtype=yb.dtype)
                    xb = torch.cat([xb, xr], dim=0)
                    yb = torch.cat([yb, yr], dim=0)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if ewc_state is not None:
                loss = loss + float(ewc_lambda) * ewc_penalty(model, ewc_state)
            loss.backward()
            opt.step()

    # Task A training
    _train_steps(X=X_a_train, y=y_a_train, steps=int(steps_a), ewc_state=None, replay=None)
    after_a = {"acc_a": _acc(model, X_a_eval, y_a_eval), "acc_b": _acc(model, X_b_eval, y_b_eval)}

    # Prepare continual state
    ewc_state = None
    if method in {"ewc", "ewc_replay"}:
        acc = EwcAccumulator()
        model.train()
        n = int(X_a_train.shape[0])
        # Fisher accumulation: sample random batches and backprop the task loss.
        for _ in range(int(fisher_batches)):
            idx = torch.randint(low=0, high=n, size=(int(batch_size),), device=device_t)
            xb = X_a_train.index_select(0, idx)
            yb = y_a_train.index_select(0, idx)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            acc.accumulate_from_grads(model)
        ewc_state = acc.finalize(model)

    replay = None
    if method in {"replay", "ewc_replay"}:
        replay = ReplayBuffer(capacity=int(replay_capacity), seed=int(seed))
        # Store some samples from A (on CPU as JSON-serializable lists).
        n = int(X_a_train.shape[0])
        take = min(int(replay_capacity) * 2, n)
        idx = torch.randperm(n, device=device_t)[:take]
        xs = X_a_train.index_select(0, idx).detach().to("cpu")
        ys = y_a_train.index_select(0, idx).detach().to("cpu")
        for row_x, row_y in zip(xs, ys):
            replay.add({"x": [float(v) for v in row_x.tolist()], "y": int(row_y.item())})

    # Task B training (continual)
    _train_steps(X=X_b_train, y=y_b_train, steps=int(steps_b), ewc_state=ewc_state, replay=replay)
    after_b = {"acc_a": _acc(model, X_a_eval, y_a_eval), "acc_b": _acc(model, X_b_eval, y_b_eval)}

    report = ContinualDemoReport(
        schema_version=1,
        settings={
            "seed": int(seed),
            "device": str(device_t),
            "method": str(method),
            "steps_a": int(steps_a),
            "steps_b": int(steps_b),
            "batch_size": int(batch_size),
            "hidden": int(hidden),
            "lr": float(lr),
            "corr": float(corr),
            "noise": float(noise),
            "n_train": int(n_train),
            "n_eval": int(n_eval),
            "ewc_lambda": float(ewc_lambda),
            "fisher_batches": int(fisher_batches),
            "replay_capacity": int(replay_capacity),
            "replay_k": int(replay_k),
            "torch_version": getattr(torch, "__version__", None),
        },
        metrics={
            "after_task_a": after_a,
            "after_task_b": after_b,
            "forgetting_acc_a": float(after_a["acc_a"] - after_b["acc_a"]),
            "gain_acc_b": float(after_b["acc_b"] - after_a["acc_b"]),
        },
    )

    if output is None:
        out_path = Path("runs") / "yolozu_demos" / "continual" / f"continual_demo_{_utc_run_id()}.json"
    else:
        out_path = Path(output)
        if out_path.is_dir() or str(output).endswith(("/", "\\")):
            out_path = out_path / f"continual_demo_{_utc_run_id()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
