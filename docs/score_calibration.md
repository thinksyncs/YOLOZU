# Score calibration (no NMS)

This repo provides a lightweight calibration helper that rescales detection scores using temperature scaling and selects the best temperature on a fixed evaluation subset.

## CLI

```bash
python3 tools/calibrate_scores.py \
  --dataset data/coco128 \
  --predictions reports/predictions_dummy.json \
  --output reports/predictions_calibrated.json \
  --output-report reports/calibration_report.json \
  --output-artifact reports/calibration_artifact.json \
  --temperatures 0.5,1.0,1.5,2.0
```

## Outputs

- `reports/predictions_calibrated.json`: calibrated predictions JSON.
- `reports/calibration_report.json`: before/after metrics + selected temperature.
- `reports/calibration_artifact.json`: grid search results.

## Notes

This uses a lightweight mAP proxy (`yolozu.simple_map`) to avoid external dependencies. The metrics are suitable for relative comparisons on a fixed subset.
