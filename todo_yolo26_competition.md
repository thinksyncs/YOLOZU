# TODO — Beat YOLO26 (COCO Detect, e2e mAP, Apache-2.0 only)

Goal: outperform **YOLO26** on **COCO Detect** using **end-to-end mAP** (NMS-free), for each size bucket:
`yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`.

Constraints:
- Keep this repo and required tooling **Apache-2.0-friendly** (no GPL/AGPL code vendored or required).
- Baseline inference can run in a separate environment (ONNX Runtime / TensorRT / custom) and only exports JSON.

## Definition of Done
- [ ] We have 5 prediction JSONs for our models (n/s/m/l/x) on COCO val2017 @ 640.
- [ ] We have 5 prediction JSONs for YOLO26 (n/s/m/l/x) generated via an Apache-2.0-friendly inference path.
- [ ] `tools/eval_suite.py` produces `reports/eval_suite.json` containing e2e COCO mAP for each file.
- [ ] `baselines/yolo26_targets.json` contains the YOLO26 e2e mAP targets we’re competing against.
- [ ] `tools/check_map_targets.py` passes for all 5 size buckets (our >= target + margin).
- [ ] `tools/print_leaderboard.py` prints a clean markdown table for sharing (mAP + delta).

## Tooling (must have)

### Datasets
- [x] Keep debug dataset under `data/coco128`.
- [x] Convert official COCO JSON to YOLO-format labels: `tools/prepare_coco_yolo.py`.
- [ ] Add a short “where to put COCO” doc snippet + example paths (under `data/`).

### Predictions JSON interoperability
- [x] Define/validate predictions schema: `tools/validate_predictions.py`.
- [x] Normalize class ids (category_id -> class_id) via `labels/<split>/classes.json`: `tools/normalize_predictions.py`.
- [x] Support bbox formats commonly produced by inference engines (already supported in eval via `--bbox-format`).
- [x] Add a “reference exporter” skeleton for ONNX Runtime / TensorRT (Apache-2.0 code only).

### Evaluation
- [x] COCO mAP evaluation (e2e = no NMS in evaluator): `tools/eval_coco.py`.
- [x] Multi-file suite evaluation: `tools/eval_suite.py`.
- [x] Gate vs target thresholds: `tools/check_map_targets.py`.
- [x] Leaderboard/summary table tool (markdown/tsv): `tools/print_leaderboard.py`.
- [ ] Repro config capture (write eval settings into output JSON: imgsz, conf, max_det, preprocessing).

## Model-side (later; requires training/inference env)
- [ ] Define YOLOZU size buckets matching YOLO26 compute (param/FLOPs envelopes).
- [ ] Implement exportable inference adapters for each bucket (ONNX + TRT).
- [ ] Train/evaluate loop that can actually beat the targets.
