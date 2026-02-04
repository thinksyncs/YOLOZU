import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.image_size import get_image_size
from yolozu.letterbox import compute_letterbox


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to ONNX model to build.")
    p.add_argument("--engine", required=True, help="Where to write TensorRT engine plan.")
    p.add_argument("--input-name", default="images", help="Input tensor name (default: images).")
    p.add_argument("--input-shape", default="1x3x640x640", help="Input shape (default: 1x3x640x640).")
    p.add_argument("--workspace-mb", type=int, default=4096, help="Workspace size in MB (default: 4096).")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 build (if supported).")
    p.add_argument("--int8", action="store_true", help="Enable INT8 build with calibration.")
    p.add_argument("--calib-dataset", default=None, help="YOLO-format dataset root for INT8 calibration.")
    p.add_argument("--calib-max-images", type=int, default=128, help="Max calibration images (default: 128).")
    p.add_argument("--calib-cache", default=None, help="Calibration cache path (optional).")
    p.add_argument("--output-meta", default=None, help="Optional JSON metadata output path.")
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in text.replace("x", " ").replace(",", " ").split()]
    if len(parts) != 4:
        raise ValueError(f"invalid input shape: {text}")
    return parts[0], parts[1], parts[2], parts[3]


def _resolve_input_name(network, requested: str) -> str:
    if not hasattr(network, "num_inputs"):
        return requested
    inputs = [network.get_input(i).name for i in range(network.num_inputs)]
    if requested in inputs:
        return requested
    if inputs:
        print(f"warning: input '{requested}' not found, using '{inputs[0]}'")
        return inputs[0]
    return requested


class _ImageCalibrator:
    def __init__(self, *, dataset_root: Path, input_size: int, input_name: str, max_images: int, cache_path: Path | None):
        if np is None:
            raise RuntimeError("numpy is required for INT8 calibration")
        try:
            import pycuda.driver as cuda  # type: ignore
            import pycuda.autoinit  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pycuda is required for INT8 calibration (pip install pycuda)") from exc

        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("opencv-python is required for calibration (pip install opencv-python)") from exc

        self._cuda = cuda
        self._cv2 = cv2
        self._input_name = input_name
        self._input_size = int(input_size)
        self._cache_path = cache_path

        manifest = build_manifest(dataset_root, split=None)
        records = manifest["images"]
        if max_images is not None:
            records = records[: int(max_images)]
        self._records = records
        self._iter = iter(self._records)

        self._batch = np.zeros((1, 3, self._input_size, self._input_size), dtype=np.float32)
        self._device = cuda.mem_alloc(self._batch.nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        try:
            record = next(self._iter)
        except StopIteration:
            return None

        image_path = record["image"]
        w, h = get_image_size(image_path)
        letterbox = compute_letterbox(orig_w=w, orig_h=h, input_size=self._input_size)

        img = self._cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"failed to load image: {image_path}")

        pad_w = float(self._input_size) - float(letterbox.new_w)
        pad_h = float(self._input_size) - float(letterbox.new_h)
        pad_x = pad_w / 2.0
        pad_y = pad_h / 2.0
        left = int(letterbox.pad_x)
        top = int(letterbox.pad_y)
        right = int(round(pad_x + 0.1))
        bottom = int(round(pad_y + 0.1))

        if (img.shape[1], img.shape[0]) != (letterbox.new_w, letterbox.new_h):
            img = self._cv2.resize(img, (letterbox.new_w, letterbox.new_h), interpolation=self._cv2.INTER_LINEAR)

        img = self._cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            self._cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        img = img[..., ::-1]  # BGR to RGB

        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        self._batch[0] = x

        self._cuda.memcpy_htod(self._device, self._batch)
        return [int(self._device)]

    def read_calibration_cache(self):
        if self._cache_path and self._cache_path.exists():
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_bytes(cache)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for TensorRT build")

    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tensorrt is required (pip install tensorrt)") from exc

    onnx_path = Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = repo_root / onnx_path
    if not onnx_path.exists():
        raise SystemExit(f"onnx model not found: {onnx_path}")

    engine_path = Path(args.engine)
    if not engine_path.is_absolute():
        engine_path = repo_root / engine_path
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    input_shape = _parse_shape(args.input_shape)

    logger = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(
        network, logger
    ) as parser:
        with onnx_path.open("rb") as f:
            if not parser.parse(f.read()):
                errors = "\n".join([parser.get_error(i).desc() for i in range(parser.num_errors)])
                raise RuntimeError(f"failed to parse ONNX:\n{errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_mb) * 1024 * 1024)

        input_name = _resolve_input_name(network, args.input_name)

        if args.fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("warning: FP16 requested but platform does not support fast FP16")

        if args.int8:
            if not builder.platform_has_fast_int8:
                raise RuntimeError("INT8 requested but platform does not support fast INT8")
            if not args.calib_dataset:
                raise SystemExit("--calib-dataset is required for INT8 builds")
            config.set_flag(trt.BuilderFlag.INT8)

            dataset_root = Path(args.calib_dataset)
            if not dataset_root.is_absolute():
                dataset_root = repo_root / dataset_root
            if not dataset_root.exists():
                raise SystemExit(f"calibration dataset not found: {dataset_root}")

            cache_path = None
            if args.calib_cache:
                cache_path = Path(args.calib_cache)
                if not cache_path.is_absolute():
                    cache_path = repo_root / cache_path

            class TRTCalibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, **kwargs):
                    super().__init__()
                    self._inner = _ImageCalibrator(**kwargs)

                def get_batch_size(self):
                    return self._inner.get_batch_size()

                def get_batch(self, names):
                    return self._inner.get_batch(names)

                def read_calibration_cache(self):
                    return self._inner.read_calibration_cache()

                def write_calibration_cache(self, cache):
                    self._inner.write_calibration_cache(cache)

            calibrator = TRTCalibrator(
                dataset_root=dataset_root,
                input_size=input_shape[2],
                input_name=args.input_name,
                max_images=int(args.calib_max_images),
                cache_path=cache_path,
            )
            config.int8_calibrator = calibrator

        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT build failed")

        engine_path.write_bytes(engine.serialize())

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "onnx": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "engine": str(engine_path),
        "engine_sha256": _sha256(engine_path),
        "input_name": input_name,
        "input_shape": "x".join([str(v) for v in input_shape]),
        "fp16": bool(args.fp16),
        "int8": bool(args.int8),
        "workspace_mb": int(args.workspace_mb),
        "calib_dataset": args.calib_dataset,
        "calib_max_images": args.calib_max_images,
        "calib_cache": args.calib_cache,
        "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
    }

    if args.output_meta:
        meta_path = Path(args.output_meta)
        if not meta_path.is_absolute():
            meta_path = repo_root / meta_path
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        print(meta_path)
    else:
        print(engine_path)


if __name__ == "__main__":
    main()
