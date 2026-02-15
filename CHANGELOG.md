# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
