# YOLOZU Spec (repo feature summary)

## Purpose
YOLOZU is a lightweight evaluation and scaffolding repo for real‑time monocular RGB detection + depth + 6DoF pose (RT‑DETR‑based). It emphasizes CPU‑minimum dev/tests (GPU optional), precomputed inference JSON ingestion, and reproducible evaluation.

## Core capabilities
### 1) Dataset I/O (YOLO‑format)
- Image layout: `images/<split>/*.jpg`
- Labels: `labels/<split>/*.txt` (YOLO: `class cx cy w h` normalized)
- Optional metadata: `labels/<split>/<image>.json`
  - Masks/seg: `mask_path` / `mask` / `M`
  - Depth: `depth_path` / `depth` / `D_obj`
  - Pose: `R_gt` / `t_gt` (or `pose`)
  - Intrinsics: `K_gt` / `intrinsics`

### 2) Mask‑only label derivation
- If YOLO txt labels are missing and a mask is provided, bbox+class can be derived from masks.
- Color mask (RGB): unique colors become classes (optionally `mask_class_map`).
- Instance mask (single‑channel IDs): non‑zero IDs become instances, class via `mask_class_id`.

### 3) Training scaffold (RT‑DETR pose)
- Minimal training loop (`rtdetr_pose/tools/train_minimal.py`).
- Optional matcher (Hungarian) with staged cost terms.
- MIM masking + teacher distillation schedules.
- Denoising target augmentation.
- Optional LoRA (Linear) for parameter-efficient finetuning (head-only by default).
- Optimizers: AdamW or SGD.
- LR warmup + schedule (`none`, `linear`, `cos`).
- Progress bar + per‑step loss logging.
- Metrics outputs: JSONL/CSV + TensorBoard logging.
- Default ONNX export at end of training.

### 4) Inference + constraints
- Constraint evaluation: depth prior, plane, upright constraints.
- Translation recovery from bbox/offsets + corrected intrinsics.
- Inference utilities for constraints + template verification.

### 5) Template verification & gating
- Symmetry‑aware template scoring utilities.
- Low‑FP gate via `score_tmp_sym`.

### 6) Predictions JSON contract
- Stable schema for evaluation ingestion.
- Supported shapes: list entries, wrapped object, or map (image -> detections).
- See `docs/predictions_schema.md` + `schemas/predictions.schema.json`.

### 7) Evaluation harness
- COCO eval conversion from YOLO labels and predictions JSON.
- NMS‑free e2e mAP evaluation.
- Scenario suite report (fps/recall/depth/pose/rejection).

### 8) Test‑time adaptation (TTA/TTT)
- TTA transforms (flip‑based) with logging (prediction-space post-transform).
- TTT utilities (MIM + Tent runners) are available as library code; integration into `tools/export_predictions.py` is planned (see `docs/ttt_integration_plan.md`).

### 9) CLI convenience
- `python -m yolozu train train_setting.yaml`
- `python -m yolozu test test_setting.yaml`

## Contracts
- **Predictions Schema**: `docs/predictions_schema.md`
- **Adapter Contract**: `docs/adapter_contract.md`

## Non‑goals
- Full production training stack (this is a scaffold).
- Heavy dependencies required for local evaluation.

## Versioning
- Internal schema versioning for predictions JSON (v1).
- Backward‑compatible additions allowed; breaking changes require version bump.
