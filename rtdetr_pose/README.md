# RT-DETR Pose Scaffolding

This folder contains a minimal PyTorch/TensorRT-oriented scaffolding for the RT-DETR 6DoF pose spec.

## Quick checks
- Dataset audit (default coco128 path): `python3 tools/dataset_audit.py`
- Dataset audit (with deeper checks): `python3 tools/dataset_audit.py --check-content --check-ranges --fail-on-issues`
- Dataset test: `python3 -m unittest tests/test_dataset.py`

## Minimal training scaffold (CPU)
- Install deps: `python3 -m pip install -r requirements-test.txt`
- Fetch coco128 (once, from repo root): `bash tools/fetch_coco128.sh`
- Run: `python3 tools/train_minimal.py --epochs 1 --batch-size 2 --max-steps 30`
- Continual FT (SDFT-inspired): add a frozen teacher checkpoint via `--self-distill-from <ckpt>` (defaults: keys=logits,bbox; logits uses reverse-KL).

Optional full-GT consumption (mask/depth)
- Derive `z` (and `t` if `K_gt` exists) from per-instance `D_obj` at bbox center: `python3 tools/train_minimal.py --use-matcher --z-from-dobj`
- If `M`/`D_obj` are stored as file paths instead of inlined arrays, allow loading: `python3 tools/train_minimal.py --use-matcher --z-from-dobj --load-aux`

Mask-only labels (simple mode)
- If YOLO labels are missing and `mask`/`mask_path` is provided, `--load-aux` will derive bbox+class from the mask.
- `mask_format: color` uses RGB colors as classes (optional `mask_class_map`).
- `mask_format: instance` treats non-zero IDs as instances (single `mask_class_id`, default 0).

Mask-only labels (optional)
- If YOLO txt labels are missing and `labels/<split>/<image>.json` has `mask_path`, the loader will derive bbox labels from the mask.
- Single PNG with multiple classes: use a single-channel mask (values are class ids) or RGB mask (unique colors become classes).
- Multiple PNGs (one per class): set `mask_path` to a list and `mask_classes` to class ids; `mask_instances: true` splits connected components into instances.

## Config
- Example config: `configs/base.json`

## Notes
- The model is a stub to wire losses/metrics and export; integrate with a full RT-DETR implementation next.
- `configs/base.json` supports `model.backbone_name` (e.g. `cspresnet`, `tiny_cnn`) and a `loss` section for swapping/tuning loss weights.
