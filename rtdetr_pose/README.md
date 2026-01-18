# RT-DETR Pose Scaffolding

This folder contains a minimal PyTorch/TensorRT-oriented scaffolding for the RT-DETR 6DoF pose spec.

## Quick checks
- Dataset audit: `python3 tools/dataset_audit.py`
- Dataset test: `python3 -m unittest tests/test_dataset.py`

## Config
- Example config: `configs/base.json`

## Notes
- The model is a stub to wire losses/metrics and export; integrate with a full RT-DETR implementation next.
