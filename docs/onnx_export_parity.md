# ONNX export parity check (predictions JSON)

Goal: ensure an ONNX export produces **numerically equivalent** predictions to a PyTorch reference (within tolerances), using the same protocol/bbox format.

## Produce two predictions JSONs

1) **Reference** (PyTorch): run your reference inference path and export YOLOZU predictions JSON.
2) **Candidate** (ONNXRuntime): run your ONNX model and export YOLOZU predictions JSON (e.g. `tools/export_predictions_onnxrt.py`).

Important:
- Use the same preprocessing (imgsz=640 + letterbox) and output format (`cxcywh_norm`).
- Do not apply NMS (e2e/no-NMS protocol).

## Run parity checker

```bash
python3 tools/check_predictions_parity.py \
  --reference /path/to/pred_ref.json \
  --candidate /path/to/pred_onnxrt.json \
  --iou-thresh 0.99 \
  --score-atol 1e-4 \
  --bbox-atol 1e-4
```

The checker exits non-zero and prints a JSON report if any image/detection mismatches.

