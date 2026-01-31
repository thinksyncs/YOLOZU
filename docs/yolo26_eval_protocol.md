# YOLO26 evaluation protocol (COCO Detect, e2e mAP)

This repository uses a pinned evaluation protocol for comparing against **YOLO26** targets.

Protocol file: `protocols/yolo26_eval.json`

## Canonical settings

- Task: `detect`
- Dataset: COCO
- Split: `val2017`
- Image size: `640` (exporter responsibility; evaluator does not resize)
- BBox format: `cxcywh_norm` (normalized `[0,1]` center-x/center-y/width/height)
- Metric key: `map50_95`
- Metric meaning: e2e mAP@[.5:.95] (bbox, no NMS)

## e2e mAP (no NMS)

“End-to-end” means **the evaluator scores detections exactly as provided**; it does not apply NMS.
If your exporter runs NMS, you are not measuring e2e performance.

## Reporting schema

Evaluation outputs are JSON and include protocol metadata:

- Single file: `reports/coco_eval.json` (from `tools/eval_coco.py`)
- Multi-file suite: `reports/eval_suite.json` (from `tools/eval_suite.py`)

Both reports include:
- `protocol_id` / `protocol`
- `dataset`, `split` (effective), `split_requested`
- `bbox_format`

## Targets

Targets live in `baselines/yolo26_targets.json` and are validated by:

`python3 tools/validate_map_targets.py --targets baselines/yolo26_targets.json`

