# Backend parity matrix automation

This command automates parity diff statistics across:

- `torch`
- `onnxrt`
- `trt`
- `opencv_dnn`
- `custom_cpp`

It generates both JSON and HTML reports in a reproducible run directory.

## One-command usage

```bash
python3 tools/backend_parity_matrix.py \
  --backend-predictions torch=reports/pred_torch.json \
  --backend-predictions onnxrt=reports/pred_onnxrt.json \
  --backend-predictions trt=reports/pred_trt.json \
  --backend-predictions opencv_dnn=reports/pred_opencv_dnn.json \
  --backend-predictions custom_cpp=reports/pred_custom_cpp.json \
  --reference-backend torch \
  --image-size 640 \
  --iou-thresh 0.99 \
  --score-atol 1e-4 \
  --bbox-atol 1e-4
```

Default outputs:

- `runs/backend_parity_matrix/<run_id>/reports/backend_parity_matrix.json`
- `runs/backend_parity_matrix/<run_id>/reports/backend_parity_matrix.html`

## What the report includes

- Per-backend parity status (`ok`/`fail`)
- Per-backend delta summary (`failure_images`, `total_failures`, `score_abs_max`, `bbox_abs_max`)
- Thresholds used for comparison
- `fixed_input_fingerprint` for reproducible reruns on fixed inputs
- `run_record` metadata

## Reproducibility

Re-run with exactly the same backend prediction files and thresholds to get the same `fixed_input_fingerprint`.
