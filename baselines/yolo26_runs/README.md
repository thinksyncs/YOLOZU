# YOLO26 baseline run archives

Use `tools/import_yolo26_baseline.py` to:
- validate externally-generated YOLO26 predictions JSON files
- run a pinned evaluation (`--protocol yolo26`)
- archive `eval_suite.json` + run metadata (commands + hardware)

Example:

```bash
python3 tools/import_yolo26_baseline.py \
  --dataset /path/to/coco-yolo \
  --predictions-glob 'reports/pred_yolo26*.json' \
  --notes 'OR/TRT on RTX 4090; exported pre-NMS logits→decoded boxes (no NMS)'
```

Archives are written to `baselines/yolo26_runs/<run-id>/`.
