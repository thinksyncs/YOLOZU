# YOLOZU (萬)

Pronunciation: Yaoyorozu (yorozu). Official ASCII name: YOLOZU.

YOLOZU is a lightweight evaluation + scaffolding repo for **real-time monocular RGB** detection + depth + 6DoF pose (RT-DETR-based).

Recommended deployment path (canonical): PyTorch → ONNX → TensorRT (TRT).

It focuses on:
- CPU-minimum dev/tests (GPU optional)
- A versioned predictions-JSON contract for evaluation
- Minimal training scaffold (RT-DETR pose) with useful defaults (metrics, progress, export)

## Feature highlights (what you can do)

- Dataset I/O: YOLO-format images/labels + optional per-image JSON metadata.
- Stable evaluation contract: versioned predictions-JSON schema + adapter contract.
- Inference/export: `tools/export_predictions.py` produces predictions JSON from adapters.
- Test-time adaptation options:
  - TTA: lightweight prediction-space post-transform (`--tta`).
  - TTT: pre-prediction test-time training (Tent or MIM) via `--ttt` (adapter + torch required).
- Evaluation: COCO mAP conversion/eval and scenario suite reporting.
- Training scaffold: minimal RT-DETR pose trainer with metrics output, ONNX export, and optional LoRA.

## Documentation

Start here: [docs/training_inference_export.md](docs/training_inference_export.md)

- Repo feature summary: [docs/yolozu_spec.md](docs/yolozu_spec.md)
- Model/spec note: [rt_detr_6dof_geom_mim_spec_en_v0_4.md](rt_detr_6dof_geom_mim_spec_en_v0_4.md)
- Training / inference / export quick steps: [docs/training_inference_export.md](docs/training_inference_export.md)
- Predictions schema (stable): [docs/predictions_schema.md](docs/predictions_schema.md)
- Adapter contract (stable): [docs/adapter_contract.md](docs/adapter_contract.md)
- License policy: [docs/license_policy.md](docs/license_policy.md)

## Quick start (coco128)

1) Install test dependencies (CPU PyTorch is OK for local dev):

```bash
python3 -m pip install -r requirements-test.txt
```

2) Fetch the tiny dataset (once):

```bash
bash tools/fetch_coco128.sh
```

3) Run a minimal check (pytest):

```bash
pytest -q
```

### GPU notes
- GPU is supported (training/inference): install CUDA-enabled PyTorch in your environment and use `--device cuda:0`.
- CI/dev does not require GPU; many checks are CPU-friendly.

## CLI (simple train/test)

Run flows with YAML settings:

```bash
python -m yolozu train train_setting.yaml
python -m yolozu test test_setting.yaml
```

Or use the wrapper:

```bash
./tools/yolozu train train_setting.yaml
./tools/yolozu test test_setting.yaml
```

Templates:
- `train_setting.yaml`
- `test_setting.yaml`

## Training scaffold (RT-DETR pose)

The minimal trainer is implemented in `rtdetr_pose/tools/train_minimal.py`.

What it provides out-of-the-box:
- Optimizers: AdamW or SGD
- LR warmup + schedule (`none`, `linear`, `cos`)
- Progress display (tqdm) + per-step loss logging
- Metrics written by default:
  - JSONL: `reports/train_metrics.jsonl`
  - CSV: `reports/train_metrics.csv`
- Optional TensorBoard logging: `--tensorboard-logdir reports/tb`

Plot a loss curve (requires matplotlib):

```bash
python3 tools/plot_metrics.py --jsonl reports/train_metrics.jsonl --out reports/train_loss.png
```

### ONNX export

ONNX export runs **by default after training**.

Control it via flags in the minimal trainer:
- `--export-onnx` / `--no-export-onnx`
- `--onnx-out <path>`
- `--onnx-opset <int>`

## Dataset format (YOLO + optional metadata)

Base dataset format:
- Images: `images/<split>/*.(jpg|png|...)`
- Labels: `labels/<split>/*.txt` (YOLO: `class cx cy w h` normalized)

Optional per-image metadata (JSON): `labels/<split>/<image>.json`
- Masks/seg: `mask_path` / `mask` / `M`
- Depth: `depth_path` / `depth` / `D_obj`
- Pose: `R_gt` / `t_gt` (or `pose`)
- Intrinsics: `K_gt` / `intrinsics`

Notes on units (pixels vs mm/m) and intrinsics coordinate frames:
- [docs/predictions_schema.md](docs/predictions_schema.md)

### Mask-only labels (seg -> bbox/class)

If YOLO txt labels are missing and a mask is provided, bbox+class can be derived from masks.
Details (including color/instance modes and multi-PNG-per-class options) are documented in:
- [rtdetr_pose/README.md](rtdetr_pose/README.md)

## Evaluation / contracts (stable)

This repo evaluates models through a stable predictions JSON format:
- Schema doc: [docs/predictions_schema.md](docs/predictions_schema.md)
- Machine-readable schema: [schemas/predictions.schema.json](schemas/predictions.schema.json)

Adapters power `tools/export_predictions.py --adapter <name>` and follow:
- [docs/adapter_contract.md](docs/adapter_contract.md)

## Precomputed predictions workflow (no torch required)

If you run real inference elsewhere (PyTorch/TensorRT/etc.), you can evaluate this repo without installing heavy deps locally.

- Export predictions (in an environment where the adapter can run):
  - `python3 tools/export_predictions.py --adapter rtdetr_pose --checkpoint /path/to.ckpt --max-images 50 --wrap --output reports/predictions.json`
  - TTA (prediction-space transform): `python3 tools/export_predictions.py --adapter rtdetr_pose --tta --tta-seed 0 --tta-flip-prob 0.5 --wrap --output reports/predictions_tta.json`
  - TTT (pre-prediction test-time training; updates model weights in-memory):
    - Tent: `python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-method tent --ttt-steps 5 --ttt-lr 1e-4 --wrap --output reports/predictions_ttt_tent.json`
    - MIM: `python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-method mim --ttt-steps 5 --ttt-mask-prob 0.6 --ttt-patch-size 16 --wrap --output reports/predictions_ttt_mim.json`
    - Optional log: add `--ttt-log-out reports/ttt_log.json`
- Validate the JSON:
  - `python3 tools/validate_predictions.py reports/predictions.json`
- Consume predictions locally:
  - `python3 tools/run_scenarios.py --adapter precomputed --predictions reports/predictions.json --max-images 50`

Supported predictions JSON shapes:
- `[{"image": "...", "detections": [...]}, ...]`
- `{ "predictions": [ ... ] }`
- `{ "000000000009.jpg": [...], "/abs/path.jpg": [...] }` (image -> detections)

Schema details:
- [docs/predictions_schema.md](docs/predictions_schema.md)

## COCO mAP (end-to-end, no NMS)

To compete on **e2e mAP** (NMS-free), evaluate detections as-is (no NMS postprocess applied).

This repo includes a COCO-style evaluator that:
- Builds COCO ground truth from YOLO-format labels
- Converts YOLOZU predictions JSON into COCO detections
- Runs COCO mAP via `pycocotools` (optional dependency)

Example (coco128 quick run):
- Export predictions (any adapter): `python3 tools/export_predictions.py --adapter dummy --max-images 50 --wrap --output reports/predictions.json`
- Evaluate mAP: `python3 tools/eval_coco.py --dataset data/coco128 --predictions reports/predictions.json --bbox-format cxcywh_norm --max-images 50`

Note:
- `--bbox-format cxcywh_norm` expects bbox dict `{cx,cy,w,h}` normalized to `[0,1]` (matching the RTDETR pose adapter bbox head).

## Training recipe (v1)

Reference recipe for external training runs (augment, multiscale, schedule, EMA):
- `docs/training_recipe_v1.md`

## Training, inference, export (quick steps)

- `docs/training_inference_export.md`

## Hyperparameter sweep harness

Run a configurable sweep and emit CSV/MD tables:
- `docs/hpo_sweep.md`

## External baselines (Apache-2.0-friendly)

This repo does **not** require (or vendor) any GPL/AGPL inference code.

To compare against external baselines (including YOLO26) while keeping this repo Apache-2.0-only:
- Run baseline inference in your own environment/implementation (ONNX Runtime / TensorRT / custom code).
- Export detections to YOLOZU predictions JSON (see schema below).
- (Optional) Normalize class ids using COCO `classes.json` mapping.
- Validate + evaluate mAP in this repo:
  - `python3 tools/validate_predictions.py reports/predictions.json`
  - `python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 --predictions reports/predictions.json --bbox-format cxcywh_norm`

Minimal predictions entry schema:
- `{"image": "/abs/or/rel/path.jpg", "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}]}`

Optional class-id normalization (when your exporter produces COCO `category_id`):
- `python3 tools/normalize_predictions.py --input reports/predictions.json --output reports/predictions_norm.json --classes data/coco-yolo/labels/val2017/classes.json --wrap`

## COCO dataset prep (official JSON -> YOLO-format)

If you have the official COCO layout (images + `annotations/instances_*.json`), you can generate YOLO-format labels:

- `python3 tools/prepare_coco_yolo.py --coco-root /path/to/coco --split val2017 --out /path/to/coco-yolo`

This creates:
- `/path/to/coco-yolo/labels/val2017/*.txt` (YOLO normalized `class cx cy w h`)
- `/path/to/coco-yolo/labels/val2017/classes.json` (category_id <-> class_id mapping)

### Dataset layout under `data/`

For local development, keep datasets under `data/`:
- Debug/smoke: `data/coco128` (already included)
- Full COCO (official): `data/coco` (your download)
- YOLO-format labels generated from official JSON: `data/coco-yolo` (your output from `tools/prepare_coco_yolo.py`)

### Size-bucket competition (yolo26n/s/m/l/x)

If you export `yolo26n/s/m/l/x` predictions as separate JSON files (e.g. `reports/pred_yolo26n.json`, ...),
you can score them together:

- Protocol details: `docs/yolo26_eval_protocol.md`
- `python3 tools/eval_suite.py --protocol yolo26 --dataset /path/to/coco-yolo --predictions-glob 'reports/pred_yolo26*.json' --output reports/eval_suite.json`
- Fill in targets: `baselines/yolo26_targets.json`
- Validate targets: `python3 tools/validate_map_targets.py --targets baselines/yolo26_targets.json`
- Check pass/fail: `python3 tools/check_map_targets.py --suite reports/eval_suite.json --targets baselines/yolo26_targets.json --key map50_95`
- Print a table: `python3 tools/print_leaderboard.py --suite reports/eval_suite.json --targets baselines/yolo26_targets.json --key map50_95`
- Archive the run (commands + hardware + suite output): `python3 tools/import_yolo26_baseline.py --dataset /path/to/coco-yolo --predictions-glob 'reports/pred_yolo26*.json'`

### Debug without `pycocotools`

If you don't have `pycocotools` installed yet, you can still validate/convert predictions on `data/coco128`:
- `python3 tools/export_predictions.py --adapter dummy --max-images 10 --wrap --output reports/predictions_dummy.json`
- `python3 tools/eval_coco.py --predictions reports/predictions_dummy.json --dry-run`

## Deployment notes
- Keep symmetry/commonsense logic in lightweight postprocess utilities, outside any inference graph export.

## License

Code in this repository is licensed under the Apache License, Version 2.0. See `LICENSE`.
