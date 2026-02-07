#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "rtdetr_pose"))

from yolozu.benchmark import measure_latency
from yolozu.run_record import build_run_record


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root / p


def _run_capture(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=repo_root, stderr=subprocess.STDOUT)
    except Exception:
        return None
    try:
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _nvidia_smi_memory() -> dict[str, Any] | None:
    out = _run_capture(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    )
    if not out:
        return None
    first = out.splitlines()[0].strip()
    parts = [p.strip() for p in first.split(",")]
    if len(parts) != 2:
        return {"raw": out}
    try:
        used = int(float(parts[0]))
        total = int(float(parts[1]))
    except Exception:
        return {"raw": out}
    return {"used_mib": used, "total_mib": total, "raw": out}


def _quantiles(values: np.ndarray, qs: Iterable[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        for q in qs:
            out[f"p{int(q)}"] = 0.0
        return out
    for q in qs:
        out[f"p{int(q)}"] = float(np.quantile(flat, q / 100.0))
    return out


def _diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {"ok": False, "reason": "shape_mismatch", "a_shape": list(a.shape), "b_shape": list(b.shape)}
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    finite = np.isfinite(diff)
    if not bool(finite.all()):
        return {
            "ok": False,
            "reason": "non_finite_diff",
            "shape": list(diff.shape),
            "non_finite": int((~finite).sum()),
        }
    q = _quantiles(diff, (50, 90, 95, 99))
    return {
        "ok": True,
        "shape": list(diff.shape),
        "max": float(diff.max()) if diff.size else 0.0,
        "mean": float(diff.mean()) if diff.size else 0.0,
        **q,
    }


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    denom = ex.sum(axis=axis, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return (ex / denom).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = 1.0 / (1.0 + np.exp(-x))
    return y.astype(np.float32)


def _derive_score_bbox(outputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    logits = outputs.get("logits")
    bbox = outputs.get("bbox")
    if logits is None or bbox is None:
        raise ValueError("outputs must contain logits and bbox")
    probs = _softmax(logits, axis=-1)
    score = probs.max(axis=-1)
    bbox_sig = _sigmoid(bbox)
    return score.astype(np.float32), bbox_sig.astype(np.float32)


def _parse_backends(value: str) -> list[str]:
    if not value:
        return ["torch", "onnxrt", "trt"]
    out = [v.strip().lower() for v in value.split(",") if v.strip()]
    if not out:
        return ["torch", "onnxrt", "trt"]
    for b in out:
        if b not in ("torch", "onnxrt", "trt"):
            raise ValueError(f"unknown backend: {b}")
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTDETRPose backend parity + benchmark suite (torch/onnxrt/trt).")
    p.add_argument("--config", required=True, help="rtdetr_pose JSON config.")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    p.add_argument("--image-size", type=int, default=320, help="Square input size (default: 320).")
    p.add_argument("--batch", type=int, default=1, help="Batch size (default: 1).")
    p.add_argument("--samples", type=int, default=2, help="Number of random samples for parity (default: 2).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")

    p.add_argument("--onnx", default=None, help="ONNX model path (required for onnxrt).")
    p.add_argument("--engine", default=None, help="TensorRT engine plan path (required for trt).")
    p.add_argument("--input-name", default="images", help="Input tensor name/binding (default: images).")

    p.add_argument("--backends", default="torch,onnxrt,trt", help="Comma-separated backends (default: torch,onnxrt,trt).")
    p.add_argument("--reference", default="torch", choices=("torch",), help="Reference backend for parity (default: torch).")

    p.add_argument("--score-atol", type=float, default=1e-4, help="Score absolute tolerance (default: 1e-4).")
    p.add_argument("--bbox-atol", type=float, default=1e-4, help="BBox absolute tolerance (default: 1e-4).")

    p.add_argument("--warmup", type=int, default=20, help="Benchmark warmup iterations (default: 20).")
    p.add_argument("--iterations", type=int, default=200, help="Benchmark iterations (default: 200).")

    p.add_argument("--output", default="reports/rtdetr_pose_backend_suite.json", help="Output JSON path.")
    p.add_argument("--dry-run", action="store_true", help="Write a report without running inference.")
    return p.parse_args(argv)


def _load_model(*, config_path: Path, checkpoint_path: Path | None, device: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for RTDETRPose torch backend") from exc

    from rtdetr_pose.config import load_config
    from rtdetr_pose.factory import build_model

    cfg = load_config(str(config_path))
    model = build_model(cfg.model).eval()
    if checkpoint_path is not None:
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            model_state = model.state_dict()
            filtered = {
                k: v for k, v in state.items() if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape
            }
            model.load_state_dict(filtered, strict=False)
        else:
            model.load_state_dict(state, strict=False)
    model.to(str(device))
    return model


def _infer_torch(model, *, x: np.ndarray, device: str) -> dict[str, np.ndarray]:
    import torch

    xt = torch.from_numpy(x).to(device=str(device), dtype=torch.float32)
    with torch.no_grad():
        out = model(xt)
    outputs: dict[str, np.ndarray] = {}
    for name in ("logits", "bbox", "log_z", "rot6d", "offsets", "k_delta"):
        t = out.get(name)
        if t is None:
            continue
        outputs[name] = t.detach().cpu().numpy()
    return outputs


@dataclass
class _OnnxRtRunner:
    session: Any
    input_name: str

    @classmethod
    def create(cls, *, onnx_path: Path, input_name: str):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime is required for onnxrt backend") from exc

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=ort.get_available_providers())
        return cls(session=session, input_name=str(input_name))

    def infer(self, x: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self.session.run(None, {self.input_name: x.astype(np.float32)})
        names = [o.name for o in self.session.get_outputs()]
        return {str(name): np.asarray(arr) for name, arr in zip(names, outputs)}


class _CudaBackend:
    def __init__(self, name: str, module):
        self.name = name
        self.module = module


def _load_cuda_backend() -> _CudaBackend:
    try:
        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore  # noqa: F401

        return _CudaBackend("pycuda", cuda)
    except Exception:
        pass
    try:
        from cuda import cudart  # type: ignore

        return _CudaBackend("cuda", cudart)
    except Exception:
        pass
    try:
        from cuda.bindings import runtime as cudart  # type: ignore

        return _CudaBackend("cuda", cudart)
    except Exception:
        raise RuntimeError("CUDA bindings not found (install pycuda or cuda-python)")


@dataclass
class _TrtRunner:
    engine: Any
    context: Any
    trt: Any
    backend: _CudaBackend
    input_name: str
    bindings: list[int]
    device_buffers: dict[str, object]
    host_buffers: dict[str, np.ndarray]
    last_shapes: dict[str, tuple[int, ...]]
    stream: Any
    binding_indices: dict[str, int]

    @classmethod
    def create(cls, *, engine_path: Path, input_name: str):
        try:
            import tensorrt as trt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("tensorrt is required for trt backend") from exc

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with engine_path.open("rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("failed to create TensorRT execution context")

        backend = _load_cuda_backend()
        bindings = [0] * engine.num_bindings
        device_buffers: dict[str, object] = {}
        host_buffers: dict[str, np.ndarray] = {}
        last_shapes: dict[str, tuple[int, ...]] = {}

        if backend.name == "pycuda":
            stream = backend.module.Stream()
        else:
            _, stream = backend.module.cudaStreamCreate()

        binding_indices = {engine.get_binding_name(i): i for i in range(engine.num_bindings)}
        if str(input_name) not in binding_indices:
            raise ValueError(f"input binding not found: {input_name}")

        return cls(
            engine=engine,
            context=context,
            trt=trt,
            backend=backend,
            input_name=str(input_name),
            bindings=bindings,
            device_buffers=device_buffers,
            host_buffers=host_buffers,
            last_shapes=last_shapes,
            stream=stream,
            binding_indices=binding_indices,
        )

    def _alloc(self, name: str, shape: tuple[int, ...], dtype):
        size = int(np.prod(shape))
        host = np.empty(size, dtype=dtype)
        nbytes = host.nbytes
        if self.backend.name == "pycuda":
            device = self.backend.module.mem_alloc(nbytes)
        else:
            err, device = self.backend.module.cudaMalloc(nbytes)
            if err != 0:
                raise RuntimeError(f"cudaMalloc failed for {name} (error {err})")
        self.host_buffers[name] = host
        self.device_buffers[name] = device
        self.last_shapes[name] = shape

    def _ensure_buffers(self, input_shape: tuple[int, ...]):
        input_idx = self.binding_indices[self.input_name]
        self.context.set_binding_shape(input_idx, input_shape)
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.context.get_binding_shape(i))
            if any(dim <= 0 for dim in shape):
                raise RuntimeError(f"binding shape unresolved for {name}: {shape}")
            dtype = self.trt.nptype(self.engine.get_binding_dtype(i))
            if name not in self.last_shapes or self.last_shapes[name] != shape:
                self._alloc(name, shape, dtype)
            self.bindings[i] = int(self.device_buffers[name])

    def infer(self, x: np.ndarray) -> dict[str, np.ndarray]:
        input_shape = tuple(x.shape)
        self._ensure_buffers(input_shape)
        input_idx = self.binding_indices[self.input_name]
        input_dtype = self.trt.nptype(self.engine.get_binding_dtype(input_idx))
        input_tensor = x.astype(input_dtype, copy=False)

        host_input = self.host_buffers[self.input_name]
        np.copyto(host_input.reshape(input_shape), input_tensor)

        if self.backend.name == "pycuda":
            self.backend.module.memcpy_htod_async(self.device_buffers[self.input_name], host_input, self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            output_names = []
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    continue
                name = self.engine.get_binding_name(i)
                output_names.append(name)
                host = self.host_buffers[name]
                self.backend.module.memcpy_dtoh_async(host, self.device_buffers[name], self.stream)
            self.stream.synchronize()
            return {name: self.host_buffers[name].reshape(self.last_shapes[name]).copy() for name in output_names}

        err = self.backend.module.cudaMemcpyAsync(
            self.device_buffers[self.input_name],
            host_input,
            host_input.nbytes,
            self.backend.module.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed (error {err})")

        ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        output_names: list[str] = []
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                continue
            name = self.engine.get_binding_name(i)
            output_names.append(name)
            host = self.host_buffers[name]
            err = self.backend.module.cudaMemcpyAsync(
                host,
                self.device_buffers[name],
                host.nbytes,
                self.backend.module.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
            if err != 0:
                raise RuntimeError(f"cudaMemcpy D2H failed for {name} (error {err})")

        err = self.backend.module.cudaStreamSynchronize(self.stream)
        if err != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed (error {err})")

        return {name: self.host_buffers[name].reshape(self.last_shapes[name]).copy() for name in output_names}


def _maybe_torch_cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _bench_backend(name: str, step, *, warmup: int, iterations: int) -> dict[str, Any]:
    mem_before = _nvidia_smi_memory()

    def wrapped_step():
        step()
        _maybe_torch_cuda_sync()

    metrics = measure_latency(warmup=int(warmup), iterations=int(iterations), step=wrapped_step)
    mem_after = _nvidia_smi_memory()
    return {"name": name, "metrics": metrics, "memory_before": mem_before, "memory_after": mem_after}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    out_path = _resolve(str(args.output))
    if out_path is None:
        raise SystemExit("--output is required")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backends = _parse_backends(str(args.backends))
    config_path = _resolve(str(args.config))
    if config_path is None or not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")

    checkpoint_path = _resolve(str(args.checkpoint)) if args.checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    onnx_path = _resolve(str(args.onnx)) if args.onnx else None
    engine_path = _resolve(str(args.engine)) if args.engine else None

    run_record = build_run_record(repo_root=repo_root, argv=sys.argv, args=vars(args))

    report: dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "meta": {
            "run_record": run_record,
            "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "config": str(config_path),
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "onnx": None if onnx_path is None else str(onnx_path),
            "engine": None if engine_path is None else str(engine_path),
            "input_name": str(args.input_name),
            "image_size": int(args.image_size),
            "batch": int(args.batch),
            "samples": int(args.samples),
            "seed": int(args.seed),
            "backends": backends,
            "dry_run": bool(args.dry_run),
        },
        "parity": {"enabled": True, "reference": "torch", "candidates": {}, "thresholds": {"score_atol": float(args.score_atol), "bbox_atol": float(args.bbox_atol)}},
        "benchmark": {"enabled": True, "results": []},
    }

    if args.dry_run:
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(out_path)
        return 0

    # Input generator (same for all backends).
    rng = np.random.default_rng(int(args.seed))
    batch = max(1, int(args.batch))
    size = max(1, int(args.image_size))
    samples = max(1, int(args.samples))
    inputs = [rng.random((batch, 3, size, size), dtype=np.float32) for _ in range(samples)]

    model = None
    if "torch" in backends:
        model = _load_model(config_path=config_path, checkpoint_path=checkpoint_path, device=str(args.device))

    onnxrt = None
    if "onnxrt" in backends:
        if onnx_path is None:
            report["parity"]["candidates"]["onnxrt"] = {"available": False, "reason": "missing_onnx"}
        elif not onnx_path.exists():
            report["parity"]["candidates"]["onnxrt"] = {"available": False, "reason": f"onnx_not_found:{onnx_path}"}
        else:
            try:
                onnxrt = _OnnxRtRunner.create(onnx_path=onnx_path, input_name=str(args.input_name))
            except Exception as exc:
                report["parity"]["candidates"]["onnxrt"] = {"available": False, "reason": f"init_failed:{exc}"}

    trt = None
    if "trt" in backends:
        if engine_path is None:
            report["parity"]["candidates"]["trt"] = {"available": False, "reason": "missing_engine"}
        elif not engine_path.exists():
            report["parity"]["candidates"]["trt"] = {"available": False, "reason": f"engine_not_found:{engine_path}"}
        else:
            try:
                trt = _TrtRunner.create(engine_path=engine_path, input_name=str(args.input_name))
            except Exception as exc:
                report["parity"]["candidates"]["trt"] = {"available": False, "reason": f"init_failed:{exc}"}

    # Parity (torch reference).
    if model is None:
        raise SystemExit("torch backend is required as reference")

    ref_outputs = [_infer_torch(model, x=x, device=str(args.device)) for x in inputs]
    ref_scores_bbox = [_derive_score_bbox(o) for o in ref_outputs]

    def compare_candidate(name: str, infer_fn):
        cand_outputs = [infer_fn(x) for x in inputs]
        cand_scores_bbox = [_derive_score_bbox(o) for o in cand_outputs]

        score_diffs = []
        bbox_diffs = []
        for (ref_score, ref_bbox), (cand_score, cand_bbox) in zip(ref_scores_bbox, cand_scores_bbox):
            score_diffs.append(_diff_stats(ref_score, cand_score))
            bbox_diffs.append(_diff_stats(ref_bbox, cand_bbox))
        score_max = max((d.get("max", 0.0) for d in score_diffs if d.get("ok")), default=0.0)
        bbox_max = max((d.get("max", 0.0) for d in bbox_diffs if d.get("ok")), default=0.0)
        passed = bool(score_max <= float(args.score_atol) and bbox_max <= float(args.bbox_atol))
        return {
            "available": True,
            "passed": passed,
            "score": {"per_sample": score_diffs, "max_over_samples": float(score_max)},
            "bbox": {"per_sample": bbox_diffs, "max_over_samples": float(bbox_max)},
        }

    if onnxrt is not None:
        try:
            report["parity"]["candidates"]["onnxrt"] = compare_candidate("onnxrt", onnxrt.infer)
        except Exception as exc:
            report["parity"]["candidates"]["onnxrt"] = {"available": False, "reason": f"infer_failed:{exc}"}

    if trt is not None:
        try:
            report["parity"]["candidates"]["trt"] = compare_candidate("trt", trt.infer)
        except Exception as exc:
            report["parity"]["candidates"]["trt"] = {"available": False, "reason": f"infer_failed:{exc}"}

    # Benchmarks.
    bench_results = []

    # Torch.
    if model is not None:
        x0 = inputs[0]

        def torch_step():
            _infer_torch(model, x=x0, device=str(args.device))

        bench_results.append(_bench_backend("torch", torch_step, warmup=int(args.warmup), iterations=int(args.iterations)))

    if onnxrt is not None:
        x0 = inputs[0]

        def onnxrt_step():
            onnxrt.infer(x0)

        bench_results.append(_bench_backend("onnxrt", onnxrt_step, warmup=int(args.warmup), iterations=int(args.iterations)))

    if trt is not None:
        x0 = inputs[0]

        def trt_step():
            trt.infer(x0)

        bench_results.append(_bench_backend("trt", trt_step, warmup=int(args.warmup), iterations=int(args.iterations)))

    report["benchmark"]["results"] = bench_results

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
