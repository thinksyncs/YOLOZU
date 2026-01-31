# Training recipe v1 (COCO detect)

This is a **reference recipe** for training a detector in an external environment.
This repo does not ship a full training pipeline yet, but it **does** provide
tools to validate/export predictions and evaluate e2e mAP.

Goal: a reproducible recipe that can generate predictions JSON which improves
over the current baseline on coco128 (quick check) and COCO val2017.

## Scope
- Task: COCO detect
- Input size: 640 (exporter responsibility)
- Evaluation: e2e mAP (no NMS in evaluator)
- BBox format: `cxcywh_norm`

## Determinism checklist
- Fix seeds: Python, NumPy, and framework RNGs.
- Set `PYTHONHASHSEED` and log it.
- Set deterministic flags (e.g., disable nondeterministic kernels, disable benchmark mode).
- Dataloader worker seeds: `base_seed + worker_id`.
- Log: git commit, dataset hash/path, training command line, framework versions, GPU/CPU.
- For PyTorch (example):
  - `torch.use_deterministic_algorithms(True)`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cuda.matmul.allow_tf32 = False`
  - `torch.backends.cudnn.allow_tf32 = False`
  - Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`

## Data + preprocessing
- Use official COCO, convert to YOLO-format labels:
  - `python3 tools/prepare_coco_yolo.py --coco-root /path/to/coco --split train2017 --out /path/to/coco-yolo`
  - `python3 tools/prepare_coco_yolo.py --coco-root /path/to/coco --split val2017 --out /path/to/coco-yolo`
- Preserve aspect ratio (letterbox) when resizing to 640.
- Normalize images to [0,1] (or your framework default); log normalization.

## Augmentation (baseline)
Always on:
- Random horizontal flip: `p=0.5`
- HSV / color jitter: small deltas (avoid extreme hue shifts)
- Random affine: translate `0.1`, scale `0.5-1.5`, rotate `+-10 deg`, mild shear

Heavy aug (first 90% of epochs, then **off** for last 10%):
- Mosaic: `p=0.5`
- Mixup: `p=0.1`
- Copy-paste: `p=0.1`

Rationale: reduce train/test gap and stabilize final epochs by disabling heavy aug.

Suggested schedule (by epoch %):
- 0% → 90%: heavy aug **on** (mosaic/mixup/copy-paste enabled)
- 90% → 100%: heavy aug **off**; keep only Always-on aug

## Multi-scale
- Random resize **per batch** in `[512, 768]` with step `32`
- Last 10% epochs: fixed `640`

## Optimizer + schedule (starting point)
- Optimizer: SGD
  - momentum `0.937`
  - weight decay `5e-4`
- Base LR: `0.01` at **global batch 64**
  - Scale linearly with batch size
- Warmup: `3-5` epochs (lr and momentum ramp)
- Schedule: cosine decay to `0.01 * base_lr`
- Gradient clip: `1.0`
- EMA: decay `0.9999`, use EMA weights for eval/export

Implementation notes:
- Exclude weight decay for bias and norm parameters.
- If using gradient accumulation, compute warmup and decay in **steps**, not epochs.

## Training duration
- COCO full: start at `300` epochs, adjust after first eval sweep
- coco128 quick check: `10-30` epochs, batch as large as possible

## Evaluation + export
Export **pre-NMS** detections and validate schema:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json
python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 \
  --predictions /path/to/predictions.json --bbox-format cxcywh_norm
```

If you export size buckets (yolo26n/s/m/l/x), run the suite:

```bash
python3 tools/eval_suite.py --protocol yolo26 --dataset /path/to/coco-yolo \
  --predictions-glob '/path/to/pred_yolo26*.json' --output reports/eval_suite.json
```

Export checklist:
- Export **pre-NMS** outputs (disable NMS or set IoU very high).
- Keep a consistent cap on detections per image (e.g., top-300 by score).
- Record the score threshold and top-k settings in the run record.

## Quick sanity check (coco128)
Use coco128 to validate the pipeline before full COCO:
- Train briefly (10-30 epochs).
- Export predictions JSON.
- Evaluate via `tools/eval_coco.py` on `data/coco128`.
- Compare mAP to your previous baseline and record delta.

Minimal external run checklist (coco128):
- (Optional) normalize class ids:
  - `python3 tools/normalize_predictions.py --input /path/to/predictions.json --output /path/to/predictions_norm.json --classes /path/to/coco-yolo/labels/val2017/classes.json --wrap`
- Validate schema:
  - `python3 tools/validate_predictions.py /path/to/predictions_norm.json`
- Evaluate mAP:
  - `python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 --predictions /path/to/predictions_norm.json --bbox-format cxcywh_norm`

Target: $\Delta$ mAP50-95 $> 0$ vs the current baseline on coco128 (even a small gain is acceptable).

## Run record (required)
Keep a short run note in your training env:
- Dataset path + hash
- Command line + config file
- Augment schedule + multiscale range
- Optimizer, LR, warmup, EMA
- Seed + determinism flags
- Framework versions + hardware

Suggested run record template:

```text
run_id: yolo26_recipe_v1_<date>
code_commit: <git sha>
dataset: /path/to/coco-yolo (hash=...)
train_cmd: ...
epochs: 300 (final 10% heavy aug off, fixed 640)
batch: 64 (global) | accum: ...
opt: SGD lr=0.01 wd=5e-4 momentum=0.937 warmup=5
ema: decay=0.9999 (used for eval/export)
multiscale: [512,768] step 32
aug: mosaic=0.5 mixup=0.1 copy-paste=0.1 flip=0.5 hsv=on
export: pre-nms topk=300 score_thresh=0.001
metrics: coco128 mAP50-95=... (baseline=..., delta=...)
predictions_json: /path/to/predictions.json
hardware: <gpu> | framework: <version>
```
