# Prediction distillation helper

This repo includes a lightweight distillation helper that blends teacher predictions into student predictions and records a distillation loss term.

## CLI

```bash
python3 tools/distill_predictions.py \
  --student reports/predictions_student.json \
  --teacher reports/predictions_teacher.json \
  --dataset data/coco128 \
  --output reports/predictions_distilled.json \
  --output-report reports/distill_report.json \
  --add-missing
```

## Config

Optional JSON config allows enabling/disabling distillation and tuning parameters:

```json
{
  "enabled": true,
  "iou_threshold": 0.7,
  "alpha": 0.5,
  "add_missing": true,
  "add_score_scale": 0.5
}
```

## Notes

Evaluation uses the lightweight `yolozu.simple_map` proxy to compare student vs distilled predictions on a fixed subset.
