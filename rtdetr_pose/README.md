# RT-DETR Pose Scaffolding

This folder contains a minimal PyTorch/TensorRT-oriented scaffolding for the RT-DETR 6DoF pose spec.

## Quick checks
- Dataset audit: `python3 tools/dataset_audit.py`
- Dataset test: `python3 -m unittest tests/test_dataset.py`

## Minimal training scaffold (CPU)
- Install deps: `python3 -m pip install -r requirements-test.txt`
- Fetch coco128 (once, from repo root): `bash tools/fetch_coco128.sh`
- Run: `python3 tools/train_minimal.py --epochs 1 --batch-size 2 --max-steps 30`

## Config
- Example config: `configs/base.json`

## Notes
- The model is a stub to wire losses/metrics and export; integrate with a full RT-DETR implementation next.
