# Runpod / Linux GPU quickstart (TensorRT pipeline)

This folder is a minimal, repo-local entrypoint for running the **YOLO26 TensorRT pipeline** on a Linux + NVIDIA GPU
machine (e.g., Runpod). It is designed so macOS users can still work on CPU-only tooling, and defer GPU work to Runpod.

## Inputs

- **YOLO-format COCO dataset root** (images + labels):
  - `.../images/<split>/*.jpg`
  - `.../labels/<split>/*.txt`
- **ONNX models** per bucket (file names are flexible; use a template):
  - `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`

## Outputs (default paths)

- TensorRT engines: `engines/{bucket}_{precision}.plan`
- Engine build metadata: `reports/trt_engine_{bucket}_{precision}.json`
- ONNXRuntime predictions: `reports/pred_onnxrt_{bucket}.json`
- TensorRT predictions: `reports/pred_trt_{bucket}.json`
- Parity reports: `reports/parity_{bucket}.json`
- COCO suite report: `reports/eval_suite_trt.json`
- Per-engine latency: `reports/latency_{bucket}.json`
- Latency summary report: `reports/benchmark_latency.json` (+ optional `reports/benchmark_latency.jsonl`)

## One-command pipeline

Run the wrapper script (it calls `tools/run_trt_pipeline.py`):

```bash
bash deploy/runpod/run_trt_pipeline.sh \
  --dataset /data/coco-yolo \
  --onnx-template /data/models/{bucket}.onnx \
  --precision fp16 \
  --combined-output output0 \
  --boxes-scale abs \
  --min-score 0.0 \
  --topk 300 \
  --max-images 500
```

Notes:
- `--combined-output output0` matches common Ultralytics ONNX exports. If your ONNX has different output names/layout,
  adjust the exporter flags.
- The pipeline supports `--dry-run` to generate schema-correct artifacts without requiring TensorRT/onnxruntime/cv2.

## Docker skeleton

TensorRT is not installed via pip; use an NVIDIA/TensorRT base image and add Python deps for YOLOZU’s tooling.

Build (example):

```bash
docker build -f deploy/runpod/Dockerfile -t yolozu:2026-02-08-trt .
```

Run (example):

```bash
docker run --gpus all -it --rm \
  -v "$PWD:/workspace/YOLOZU" \
  -v "/data:/data" \
  yolozu:2026-02-08-trt
```

Then run the pipeline command above inside the container.

## Compliance artifacts (doctor + dependency licenses)

Inside the container, you can emit reproducibility/compliance artifacts:

```bash
bash deploy/runpod/run_compliance.sh
```

This writes:
- `reports/doctor.json`
- `reports/dependency_licenses.json`
