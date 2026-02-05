# Adapter contract (v1)

Adapters power `tools/export_predictions.py --adapter <name>`.

## Required behavior
An adapter must implement:
- `predict(records: list[dict]) -> list[dict]`
  - Each entry must be `{image, detections}`
  - Detections must include `class_id`, `score`, and `bbox` (`cx,cy,w,h`)

## Optional behavior
- `predict` may include extra keys per detection (mask/depth/pose/intrinsics)
- `records` are built from YOLO-format datasets via `yolozu.dataset.build_manifest`

## Stability
- `predict` signature and output schema are **stable**.
- New optional fields may be added without breaking old clients.

## Versioning
Adapters should be compatible with predictions schema `v1`.
