# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- FRACAL calibration now supports both bbox and instance-segmentation predictions via `yolozu calibrate --task {bbox,seg,auto}`.
- FRACAL class-frequency stats can now be exported/reused through `--stats-out` and `--stats-in`, enabling stable calibration across runs.
- Trainer now emits FRACAL stats from training records via `--fracal-stats-out`; with `--run-contract`, default output is `runs/<run-id>/reports/fracal_stats_bbox.json`.
- Added alternative calibration methods in `yolozu calibrate`: Logit Adjustment (`--method la --tau`) and NorCal (`--method norcal --gamma`) for side-by-side comparison with FRACAL.
- Added temperature scaling in `yolozu calibrate` (`--method temperature --temperature`, optional `--fit-temperature` with `--temperature-grid`).

### Tests
- Added regression coverage for FRACAL stats reuse and instance-segmentation calibration behavior.
- Added run-contract default-path coverage for FRACAL stats artifact output.

## [0.1.2] - 2026-02-17

### Added
- COCO/Detectron2 keypoint schema ingest on dataset import: `categories[].keypoints` and `categories[].skeleton` are persisted into wrapper metadata (`dataset.json` and `labels/<split>/classes.json`).
- RT-DETR pose trainer auto keypoint setup from dataset metadata: when `--num-keypoints` is not provided, it is inferred from imported keypoint schema.
- Horizontal flip keypoint pairing support based on left/right keypoint names to keep keypoint semantics consistent during augmentation.

### Tests
- Added regression coverage for keypoint schema import persistence and trainer keypoint flip-pair derivation.

## [0.1.1] - 2026-02-15

### Added
- `yolozu validate dataset` to sanity-check YOLO-format datasets (images/labels + normalized bbox ranges).
- `yolozu demo continual --compare/--methods` to run a multi-method continual-learning demo suite and optionally emit a markdown table (`--markdown`).

## [0.1.0] - 2026-02-15

Initial OSS release.

### Added
- `yolozu` pip CLI: `doctor`, `export`, `validate`, `eval-instance-seg`, `resources`, `demo`.
- Predictions JSON schema + validators (backend-agnostic evaluation contract).
- Instance segmentation evaluation (PNG mask contract; mask mAP + optional HTML/overlays).
- Optional extras: `yolozu[demo]` (torch), `yolozu[onnxrt]`, `yolozu[coco]`, `yolozu[full]`.
- TensorRT / ONNXRuntime pipeline helpers (repo checkout; GPU optional).
- RT-DETR pose scaffold (`rtdetr_pose/`) with minimal training + ONNX export hooks.
