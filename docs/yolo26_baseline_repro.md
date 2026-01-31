# Reproducing YOLO26 baseline predictions

This repo does not ship (or require) any non-Apache inference code. To reproduce YOLO26 baselines, run inference externally and export **YOLOZU predictions JSON** for each bucket.

Protocol reference: `docs/yolo26_eval_protocol.md`

## 1) Export predictions (external env)

Create five files (names are conventional; any names are fine as long as they match your glob):

- `reports/pred_yolo26n.json`
- `reports/pred_yolo26s.json`
- `reports/pred_yolo26m.json`
- `reports/pred_yolo26l.json`
- `reports/pred_yolo26x.json`

Each file must follow the schema validated by:

```bash
python3 tools/validate_predictions.py reports/pred_yolo26n.json
```

Important: export **pre-NMS** detections if you want true e2e/no-NMS evaluation.

### Exporter skeletons (this repo)

This repo includes Apache-2.0-friendly exporter skeletons you can adapt in your inference environment:

```bash
python3 tools/export_predictions_onnxrt.py --dataset /path/to/coco-yolo --onnx /path/to/model.onnx --wrap --output reports/pred_yolo26n.json
python3 tools/export_predictions_trt.py --dataset /path/to/coco-yolo --engine /path/to/model.plan --wrap --output reports/pred_yolo26n.json
```

Both tools are **NMS-free** (they never run NMS), and they capture run metadata when `--wrap` is enabled.

## 2) Evaluate + archive in this repo

Run a pinned evaluation and archive the suite JSON plus run metadata (commands + machine info):

```bash
python3 tools/import_yolo26_baseline.py \
  --dataset /path/to/coco-yolo \
  --predictions-glob 'reports/pred_yolo26*.json' \
  --notes 'Describe exporter, postprocess, and hardware here'
```

Outputs:
- `reports/eval_suite.json` (ignored by git)
- `baselines/yolo26_runs/<run-id>/eval_suite.json` (tracked)
- `baselines/yolo26_runs/<run-id>/run.json` (tracked)

## 3) Gate vs targets (optional)

```bash
python3 tools/check_map_targets.py \
  --suite reports/eval_suite.json \
  --targets baselines/yolo26_targets.json \
  --key map50_95
```
