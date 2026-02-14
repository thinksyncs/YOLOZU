# YOLOZU (萬)

Pronunciation: Yaoyorozu (yorozu). Official ASCII name: YOLOZU.

YOLOZU is an Apache-2.0-only, **contract-first evaluation + tooling harness** for:
- real-time monocular RGB **detection**
- monocular **depth + 6DoF pose** heads (RT-DETR-based scaffold)
- **semantic segmentation** utilities (dataset prep + mIoU evaluation)
- **instance segmentation** utilities (PNG-mask contract + mask mAP evaluation)

Recommended deployment path (canonical): PyTorch → ONNX → TensorRT (TRT).

It focuses on:
- CPU-minimum dev/tests (GPU optional)
- A stable predictions-JSON contract for evaluation (bring-your-own inference backend)
- Minimal training scaffold (RT-DETR pose) with reproducible artifacts
- Hessian-based refinement for regression head predictions (depth, rotation, offsets)

## Why YOLOZU (what’s “sellable”)

- **Backend-agnostic evaluation**: run inference in PyTorch / ONNXRuntime / TensorRT / C++ / Rust → export the same `predictions.json` → compare apples-to-apples.
- **Unified CLI**: `python3 tools/yolozu.py` wraps backends with consistent args, caching (`--cache`), and always writes run metadata (git SHA / env / GPU / config hash).
- **Parity + benchmarks**: backend diff stats (torch vs onnxrt vs trt) and fixed-protocol latency/FPS reports.
- **Safe test-time training (Tent)**: norm-only updates with guard rails (non-finite/loss/update-norm stops + rollback) and reset policies.
- **AI-friendly repo surface**: stable schemas + `tools/manifest.json` for tool discovery / automation.

## Feature highlights (what you can do)

- Dataset I/O: YOLO-format images/labels + optional per-image JSON metadata.
- Stable evaluation contract: versioned predictions-JSON schema + adapter contract.
- Unified CLI: `python3 tools/yolozu.py` (`doctor`, `export`, `predict-images`, `sweep`) for research/eval workflows.
- Inference/export: `tools/export_predictions.py` (torch adapter), `tools/export_predictions_onnxrt.py`, `tools/export_predictions_trt.py`.
- Test-time adaptation options:
  - TTA: lightweight prediction-space post-transform (`--tta`).
  - TTT: pre-prediction test-time training (Tent or MIM) via `--ttt` (adapter + torch required).
- Hessian solver: per-detection iterative refinement of regression outputs (depth, rotation, offsets) using Gauss-Newton optimization.
- Evaluation: COCO mAP conversion/eval and scenario suite reporting.
- Keypoints: YOLO pose-style keypoints in labels/predictions + PCK evaluation + optional COCO OKS mAP (`tools/eval_keypoints.py --oks`), plus parity/benchmark helpers.
- Semantic seg: dataset prep helpers + `tools/eval_segmentation.py` (mIoU/per-class IoU/ignore_index + optional HTML overlays).
- Instance seg: `tools/eval_instance_segmentation.py` (mask mAP from per-instance binary PNG masks + optional HTML overlays).
- Training scaffold: minimal RT-DETR pose trainer with metrics output, ONNX export, and optional SDFT-style self-distillation.

## Instance segmentation (PNG masks)

YOLOZU evaluates instance segmentation using **per-instance binary PNG masks** (no RLE/polygons required).

Predictions JSON (minimal):
```json
[
  {
    "image": "000001.png",
    "instances": [
      { "class_id": 0, "score": 0.9, "mask": "masks/000001_inst0.png" }
    ]
  }
]
```

Validate an artifact:
```bash
python3 tools/validate_instance_segmentation_predictions.py reports/instance_seg_predictions.json
```

Eval outputs:
- mask mAP (`map50`, `map50_95`)
- per-class AP table
- per-image diagnostics (TP/FP/FN, mean IoU) and overlay selection (`--overlay-sort {worst,best,first}`; default: `worst`)

Run the synthetic demo and render overlays/HTML:
```bash
python3 tools/eval_instance_segmentation.py \
  --dataset examples/instance_seg_demo/dataset \
  --split val2017 \
  --predictions examples/instance_seg_demo/predictions/instance_seg_predictions.json \
  --pred-root examples/instance_seg_demo/predictions \
  --classes examples/instance_seg_demo/classes.txt \
  --html reports/instance_seg_demo_eval.html \
  --overlays-dir reports/instance_seg_demo_overlays \
  --max-overlays 10
```

Same via the unified CLI:
```bash
python3 tools/yolozu.py eval-instance-seg --dataset examples/instance_seg_demo/dataset --split val2017 --predictions examples/instance_seg_demo/predictions/instance_seg_predictions.json --pred-root examples/instance_seg_demo/predictions --classes examples/instance_seg_demo/classes.txt --html reports/instance_seg_demo_eval.html --overlays-dir reports/instance_seg_demo_overlays --max-overlays 10
```

Optional: prepare COCO instance-seg dataset with per-instance PNG masks (requires `pycocotools`):
```bash
python3 tools/prepare_coco_instance_seg.py --coco-root /path/to/coco --split val2017 --out data/coco-instance-seg
```

Optional: convert COCO instance-seg predictions (RLE/polygons) into YOLOZU PNG masks (requires `pycocotools`):
```bash
python3 tools/convert_coco_instance_seg_predictions.py \
  --predictions /path/to/coco_instance_seg_preds.json \
  --instances-json /path/to/instances_val2017.json \
  --output reports/instance_seg_predictions.json \
  --masks-dir reports/instance_seg_masks
```

## Documentation

Start here: [docs/training_inference_export.md](docs/training_inference_export.md)

- Repo feature summary: [docs/yolozu_spec.md](docs/yolozu_spec.md)
- Model/spec note: [rt_detr_6dof_geom_mim_spec_en_v0_4.md](rt_detr_6dof_geom_mim_spec_en_v0_4.md)
- Training / inference / export quick steps: [docs/training_inference_export.md](docs/training_inference_export.md)
- Hessian solver for regression refinement: [docs/hessian_solver.md](docs/hessian_solver.md)
- Predictions schema (stable): [docs/predictions_schema.md](docs/predictions_schema.md)
- Adapter contract (stable): [docs/adapter_contract.md](docs/adapter_contract.md)
- License policy: [docs/license_policy.md](docs/license_policy.md)
- Tools index (AI-friendly): [docs/tools_index.md](docs/tools_index.md) / [tools/manifest.json](tools/manifest.json)

## Roadmap (priorities)

- P0 (done): Unified CLI (`torch` / `onnxruntime` / `tensorrt`) with consistent args + same output schema; always write meta (git SHA / env / GPU / seed / config hash); keep `tools/manifest.json` updated.
- P1 (done): `doctor` (deps/GPU/driver/onnxrt/TRT diagnostics) + `predict-images` (folder input → predictions JSON + overlays) + HTML report.
- P2 (partial): cache/re-run (fingerprinted runs) + sweeps (wrapper exists; expand sweeps for TTT/threshold/gate weights) + production inference cores (C++/Rust) as needed.

## Pros / Cons (project-level)

### Pros
- Apache-2.0-only utilities and evaluation harnesses (no vendored GPL/AGPL inference code).
- CPU-first development workflow: dataset tooling, validators, scenario suite, and unit tests run without a GPU.
- Adapter interface decouples inference backend from evaluation (PyTorch/ONNXRuntime/TensorRT/custom), so you can
  run inference elsewhere and still score/compare locally.
- Reproducible artifacts: stable JSON reports + optional JSONL history for regressions.
- Symmetry + commonsense constraints are treated as first-class, test-covered utilities (not ad-hoc postprocess).

### Cons / Limitations
- Not a turnkey training repo: the in-repo `rtdetr_pose/` model is scaffolding to wire data/losses/metrics/export.
  It is not expected to be competitive without significant upgrades.
- No “one command” real-time inference app is shipped here. The intended flow is:
  bring-your-own inference backend → export predictions JSON → run evaluation/scenarios in this repo.
- TensorRT development is **not** macOS-friendly: engine build/export steps assume an NVIDIA stack (typically Linux).
  On macOS you can still do CPU-side validation and keep GPU steps for Runpod/remote.
- Backend parity is fragile: preprocessing (letterbox/RGB order), output layouts, and score calibration can dominate
  mAP/FPS differences more than the model itself if they drift.
- Some tools intentionally use lightweight metrics (e.g. `yolozu.simple_map`) to avoid heavy deps; full COCOeval
  requires optional dependencies and the proper COCO layout.
- Large model weights/datasets are intentionally kept out of git; you need external storage and reproducible pointers.

## Quick start (coco128)

1) Install test dependencies (CPU PyTorch is OK for local dev):

```bash
python3 -m pip install -r requirements-test.txt
```

## Install (pip) + demos (CPU)

For development (editable install):

```bash
python3 -m pip install -e .
yolozu --help
```

Run a minimal CPU-only demo (no torch required):

```bash
yolozu demo instance-seg
```

Run a continual-learning + domain-shift demo (CPU torch required):

```bash
python3 -m pip install -e '.[demo]'
yolozu demo continual --method ewc_replay
```

## Container images (GHCR)

YOLOZU can publish Docker images to GitHub Container Registry (GHCR) on tags `vX.Y.Z`.

- Minimal (no torch): `ghcr.io/<owner>/yolozu:<tag>`
- Demo (includes torch): `ghcr.io/<owner>/yolozu-demo:<tag>`

Examples:

```bash
docker run --rm ghcr.io/<owner>/yolozu:0.1.0 doctor --output -
docker run --rm ghcr.io/<owner>/yolozu-demo:0.1.0 demo continual --method ewc_replay
```

2) Fetch the tiny dataset (once):

```bash
bash tools/fetch_coco128.sh
```

3) Run a minimal check (pytest):

```bash
pytest -q
```

Or:

```bash
python3 -m unittest -q
```

### GPU notes
- GPU is supported (training/inference): install CUDA-enabled PyTorch in your environment and use `--device cuda:0`.
- CI/dev does not require GPU; many checks are CPU-friendly.

## CLI (simple train/test)

Run flows with YAML settings:

```bash
yolozu train train_setting.yaml
yolozu test test_setting.yaml

# Equivalent:
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

Recommended usage is to set `--run-dir`, which writes a standard, reproducible artifact set:
- `metrics.jsonl` (+ final `metrics.json` / `metrics.csv`)
- `checkpoint.pt` (+ optional `checkpoint_bundle.pt`)
- `model.onnx` (+ `model.onnx.meta.json`)
- `run_record.json` (git SHA / platform / args)

Plot a loss curve (requires matplotlib):

```bash
python3 tools/plot_metrics.py --jsonl runs/<run>/metrics.jsonl --out reports/train_loss.png
```

### ONNX export

ONNX export runs when `--run-dir` is set (defaulting to `<run-dir>/model.onnx`) or when `--onnx-out` is provided.

Useful flags:
- `--run-dir <dir>`
- `--onnx-out <path>`
- `--onnx-meta-out <path>`
- `--onnx-opset <int>`
- `--onnx-dynamic-hw` (dynamic H/W axes)

## Dataset format (YOLO + optional metadata)

Base dataset format:
- Images: `images/<split>/*.(jpg|png|...)`
- Labels: `labels/<split>/*.txt` (YOLO: `class cx cy w h` normalized)

Optional per-image metadata (JSON): `labels/<split>/<image>.json`
- Masks/seg: `mask_path` / `mask` / `M`
- Depth: `depth_path` / `depth` / `D_obj`
- Pose: `R_gt` / `t_gt` (or `pose`)
- Intrinsics: `K_gt` / `intrinsics` (also supports OpenCV FileStorage-style `camera_matrix: {rows, cols, data:[...]}`)

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
  - TTA (post-transform): `python3 tools/export_predictions.py --adapter rtdetr_pose --tta --tta-seed 0 --tta-flip-prob 0.5 --wrap --output reports/predictions_tta.json`
  - TTT (pre-prediction test-time training; updates model weights in-memory):
    - Tent (safe preset + guard rails): `python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-preset safe --ttt-reset sample --wrap --output reports/predictions_ttt_safe.json`
    - MIM (safe preset + guard rails): `python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-preset mim_safe --ttt-reset sample --wrap --output reports/predictions_ttt_mim_safe.json`
    - Optional log: add `--ttt-log-out reports/ttt_log.json`
    - Recommended protocol: [docs/ttt_protocol.md](docs/ttt_protocol.md)
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

## Latency/FPS benchmark harness

Report latency/FPS per YOLO26 bucket and archive runs over time:
- `docs/benchmark_latency.md`

## Inference-time gating / score fusion

Fuse detection/template/uncertainty signals into a single score and tune weights offline (CPU-only):
- `docs/gate_weight_tuning.md`

## TensorRT FP16/INT8 pipeline

Reproducible engine build + parity validation steps:
- `docs/tensorrt_pipeline.md`

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
