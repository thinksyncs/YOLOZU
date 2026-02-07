# C++ inference template (submodule-ready)

This folder is a **minimal CMake-based C++ template** meant to help you start a production-grade inference path while keeping
YOLOZU's evaluation pipeline unchanged.

It is intentionally structured so you can:

1) develop it *in-tree* under `examples/infer_cpp/`, then
2) **extract it into a separate repo**, and
3) add it back to YOLOZU as a git submodule (once the repo URL exists).

## What it provides

- `yolozu_infer_stub`: builds everywhere, emits schema-correct `predictions.json` with empty detections
- Optional (when deps exist):
  - `yolozu_infer_onnxrt`: ONNXRuntime C++ runner (CPU)
  - `yolozu_infer_trt`: TensorRT C++ runner (Linux+NVIDIA)

All runners aim to produce YOLOZU predictions JSON so you can:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json --strict
python3 tools/eval_suite.py --predictions /path/to/predictions.json --dataset /path/to/coco-yolo
```

## Build (stub only)

```bash
cmake -S . -B build
cmake --build build -j
./build/yolozu_infer_stub --image /abs/path.jpg --output /tmp/predictions.json
```

## Build (ONNXRuntime)

This template expects ONNXRuntime headers/libs to be available and passed via `ONNXRUNTIME_ROOT`.

Example (after downloading an ONNXRuntime release tarball and extracting to `/opt/onnxruntime`):

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime
cmake --build build -j
```

## Build (TensorRT)

TensorRT C++ is easiest inside an NVIDIA/TensorRT base image (Runpod/Linux). The CMake checks for `NvInfer.h` + `libnvinfer`.

## Extracting into a submodule (recommended later)

Once you create a separate repo (e.g. `yolozu-infer-cpp`) under the same GitHub org:

1) Move this folder into its own repo (keep Apache-2.0 compatible code).
2) In YOLOZU:

```bash
git submodule add <YOUR_REPO_URL> external/yolozu-infer-cpp
```

Then update docs to point users to that submodule for production inference.

