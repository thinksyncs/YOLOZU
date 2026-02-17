# YOLOZU (萬)

日本語: [`Readme_jp.md`](Readme_jp.md)

[![CI](https://github.com/thinksyncs/YOLOZU/actions/workflows/ci.yml/badge.svg)](https://github.com/thinksyncs/YOLOZU/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/yolozu?logo=pypi&logoColor=white)](https://pypi.org/project/yolozu/)
[![Python >=3.10](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://pypi.org/project/yolozu/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

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

## Quickstart (pip users)

```bash
python3 -m pip install yolozu
yolozu doctor --output -
yolozu predict-images --backend dummy --input-dir /path/to/images
yolozu demo instance-seg
```

Optional extras:

```bash
python3 -m pip install 'yolozu[demo]'    # torch demos (CPU OK)
python3 -m pip install 'yolozu[onnxrt]'  # ONNXRuntime CPU exporter
python3 -m pip install 'yolozu[coco]'    # pycocotools COCOeval
python3 -m pip install 'yolozu[full]'
```

Docs index (start here): [`docs/README.md`](docs/README.md)

## Why YOLOZU (what’s “sellable”)

- **Bring-your-own inference + contract-first evaluation**: run inference in PyTorch / ONNXRuntime / TensorRT / C++ / Rust
  → export the same `predictions.json` → compare apples-to-apples.
- **Safe TTT (test-time training)**: presets + guard rails + reset policies (see `docs/ttt_protocol.md`).
- **Apache-2.0-only ops**: license policy + checks to keep the toolchain clean (see `docs/license_policy.md`).
- **Unified CLI**: `yolozu` (pip) + `python3 tools/yolozu.py` (repo) wrap backends with consistent args, caching (`--cache`),
  and always write run metadata (git SHA / env / GPU / config hash).
- **Parity + benchmarks**: backend diff stats (torch vs onnxrt vs trt) and fixed-protocol latency/FPS reports.
- **AI-friendly repo surface**: stable schemas + `tools/manifest.json` for tool discovery / automation.

## Feature highlights (what you can do)

- Dataset I/O: YOLO-format images/labels + optional per-image JSON metadata.
- Stable evaluation contract: versioned predictions-JSON schema + adapter contract.
- Unified CLI:
  - pip: `yolozu` (install-safe commands + CPU demos)
  - repo: `python3 tools/yolozu.py` (power-user research/eval workflows)
- Inference/export: `python3 tools/yolozu.py export --backend {torch,onnxrt,trt}` (wrapper) or the low-level scripts
  (`tools/export_predictions*.py`).
- Test-time adaptation options:
  - TTA: lightweight prediction-space post-transform (`--tta`).
  - TTT: pre-prediction test-time training (Tent or MIM) via `--ttt` on the **torch backend**
    (see `docs/ttt_protocol.md`).
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
- Model/spec note: [docs/specs/rt_detr_6dof_geom_mim_spec_en_v0_4.md](docs/specs/rt_detr_6dof_geom_mim_spec_en_v0_4.md)
- Training / inference / export quick steps: [docs/training_inference_export.md](docs/training_inference_export.md)
- Hessian solver for regression refinement: [docs/hessian_solver.md](docs/hessian_solver.md)
- Predictions schema (stable): [docs/predictions_schema.md](docs/predictions_schema.md)
- Adapter contract (stable): [docs/adapter_contract.md](docs/adapter_contract.md)
- Migration helpers: [docs/migrate.md](docs/migrate.md)
- License policy: [docs/license_policy.md](docs/license_policy.md)
- Tools index (AI-friendly): [docs/tools_index.md](docs/tools_index.md) / [tools/manifest.json](tools/manifest.json)
- AI-first usage guide: [docs/ai_first.md](docs/ai_first.md)
- PyInstaller/PyArmor packaging notes: [deploy/pyinstaller/README.md](deploy/pyinstaller/README.md)

## Roadmap (priorities)

- P0: Unified CLI (`torch` / `onnxruntime` / `tensorrt`) with consistent args + same output schema; always write meta (git SHA / env / GPU / seed / config hash); keep `tools/manifest.json` updated.
- P1: `doctor` (deps/GPU/driver/onnxrt/TRT diagnostics) + `predict-images` (folder input → predictions JSON + overlays) + HTML report.
- P2: cache/re-run (fingerprinted runs) + sweeps (wrapper exists; expand sweeps for TTT/threshold/gate weights) + production inference cores (C++/Rust) as needed.
- Long-form notes: `docs/roadmap.md`

### Status snapshot (2026-02-17)

- P0: implemented in unified wrapper CLI (`python3 tools/yolozu.py export --backend {dummy,torch,onnxrt,trt}`) with wrapped predictions JSON and `meta.run` (`git/env/gpu/seed/config_hash`).
- P1: implemented (`doctor`, `predict-images`, HTML overlays/report path).
- P2: implemented baseline (`--cache`, sweep wrapper) and ongoing expansion for broader production cores/tuning presets.

Recent compatibility additions:
- import/doctor auto-detection: `yolozu import ... --from auto`, `yolozu doctor import --config-from auto|--dataset-from auto`.
- train shorthand preview: `yolozu train --import auto --cfg <args_or_config>` writes resolved canonical `TrainConfig`.

## Pros / Operational Notes (project-level)

### Pros
- Apache-2.0-only utilities and evaluation harnesses (no vendored GPL/AGPL inference code).
- CPU-first development workflow: dataset tooling, validators, scenario suite, and unit tests run without a GPU.
- Adapter interface decouples inference backend from evaluation (PyTorch/ONNXRuntime/TensorRT/custom), so you can
  run inference elsewhere and still score/compare locally.
- Reproducible artifacts: stable JSON reports + optional JSONL history for regressions.
- Symmetry + commonsense constraints are treated as first-class, test-covered utilities (not ad-hoc postprocess).

### Operational notes and mitigations
- Training remains scaffold-first in `rtdetr_pose/` (data/loss/export wiring), while continual-learning behavior is
  immediately testable from pip with `yolozu demo continual --compare --markdown` and source training stays available via
  `yolozu train <config>` (`docs/training_inference_export.md`, requires `yolozu[train]`).
- A one-command folder inference path is available from pip: `yolozu predict-images --backend onnxrt --input-dir <dir> --onnx <model.onnx>`,
  which writes predictions JSON + overlays + HTML in one run.
- TensorRT remains NVIDIA/Linux-centric, while macOS can run CPU validation and ONNXRuntime export:
  `yolozu onnxrt export ...`; GPU/TRT build/eval is pinned to Runpod/container workflows (`docs/tensorrt_pipeline.md`).
- Backend parity drift is handled by a dedicated checker:
  `yolozu parity --reference reports/pred_torch.json --candidate reports/pred_onnxrt.json`
  plus protocol-pinned eval settings (`docs/yolo26_eval_protocol.md`).
- Lightweight metrics stay available for fast loops, and full COCOeval is directly exposed from pip:
  `python3 -m pip install 'yolozu[coco]'` then
  `yolozu eval-coco --dataset <yolo-dataset> --predictions <predictions.json>`.
- Long-tail focused post-hoc path is available without retraining:
  `yolozu calibrate --method fracal --dataset <yolo-dataset> --predictions <predictions.json>` then
  `yolozu eval-long-tail --dataset <yolo-dataset> --predictions <calibrated_predictions.json>`.
- Model weights/datasets stay outside git by design; reproducibility is maintained through stable JSON artifacts and
  pinned path conventions documented in `docs/external_inference.md` and `docs/yolo26_inference_adapters.md`.

## Install (pip users)

```bash
python3 -m pip install yolozu
yolozu --help
yolozu doctor --output -
```

Optional extras (recommended as needed):

```bash
python3 -m pip install 'yolozu[demo]'    # torch demos (CPU OK)
python3 -m pip install 'yolozu[onnxrt]'  # ONNXRuntime CPU exporter
python3 -m pip install 'yolozu[coco]'    # pycocotools COCOeval
python3 -m pip install 'yolozu[full]'
```

CPU demos:

```bash
yolozu demo instance-seg
yolozu demo continual --method ewc_replay     # requires yolozu[demo]
yolozu demo continual --compare --markdown    # suite: naive/ewc/replay/ewc_replay
```

## Source checkout (repo users)

This path unlocks the full repo tooling (`tools/`, `rtdetr_pose/`, scenarios, etc.).

```bash
python3 -m pip install -r requirements-test.txt
python3 -m pip install -e .

# Tiny smoke dataset (optional but useful for scenario runs)
bash tools/fetch_coco128.sh

python3 -m unittest -q
```

## CLI: pip vs repo

| Capability | pip (`pip install yolozu`) | repo checkout |
|---|---|---|
| Environment report | `yolozu doctor --output -` | `python3 tools/yolozu.py doctor --output reports/doctor.json` |
| Export smoke (no inference) | `yolozu export --backend labels --dataset /path/to/yolo --output reports/predictions.json --force` | same |
| Folder inference + overlays/HTML | `yolozu predict-images --backend onnxrt --input-dir /path/to/images --onnx /path/to/model.onnx` | `python3 tools/yolozu.py predict-images ...` |
| Backend parity check | `yolozu parity --reference reports/pred_torch.json --candidate reports/pred_onnxrt.json` | `python3 tools/check_predictions_parity.py ...` |
| Validate dataset layout | `yolozu validate dataset /path/to/yolo --strict` | `python3 tools/validate_dataset.py /path/to/yolo --strict` |
| Validate predictions JSON | `yolozu validate predictions reports/predictions.json --strict` | `python3 tools/validate_predictions.py reports/predictions.json --strict` |
| COCOeval mAP | `yolozu eval-coco --dataset /path/to/yolo --predictions reports/predictions.json` (requires `yolozu[coco]`) | `python3 tools/eval_coco.py ...` |
| Long-tail post-hoc + report | `yolozu calibrate --method fracal --dataset /path/to/yolo --predictions reports/predictions.json --output reports/predictions_calibrated.json && yolozu eval-long-tail --dataset /path/to/yolo --predictions reports/predictions_calibrated.json --output reports/long_tail_eval.json` | (same via `python3 tools/yolozu.py ...`) |
| Long-tail train recipe (decoupled + plugins) | `yolozu long-tail-recipe --dataset /path/to/yolo --output reports/long_tail_recipe.json --rebalance-sampler class_balanced --loss-plugin focal --logit-adjustment-tau 1.0 --lort-tau 0.3` | (same via `python3 tools/yolozu.py ...`) |
| Instance-seg eval (PNG masks) | `yolozu eval-instance-seg --dataset /path --predictions preds.json --output reports/instance_seg_eval.json` | `python3 tools/eval_instance_segmentation.py ...` |
| ONNXRuntime CPU export | `yolozu onnxrt export ...` (requires `yolozu[onnxrt]`) | `python3 tools/export_predictions_onnxrt.py ...` |
| Training scaffold | `yolozu train configs/examples/train_contract.yaml --run-id exp01` (requires `yolozu[train]`) | `python3 rtdetr_pose/tools/train_minimal.py ...` |
| Scenario suite | `yolozu test configs/examples/test_setting.yaml` | `python3 tools/run_scenarios.py ...` |

The “power-user” unified CLI lives in-repo: `python3 tools/yolozu.py --help`.

Path behavior in tool CLIs:
- Relative input paths are resolved from the current working directory (with repo-root fallback for compatibility).
- Relative output paths are written under the current working directory.
- For config-driven tools such as `tools/tune_gate_weights.py`, relative paths in the config are resolved from the config file directory.

## Container images (GHCR)

YOLOZU can publish Docker images to GitHub Container Registry (GHCR) on tags `vX.Y.Z`.

- Minimal (no torch): `ghcr.io/<owner>/yolozu:<tag>`
- Demo (includes torch): `ghcr.io/<owner>/yolozu-demo:<tag>`

Examples:

```bash
docker run --rm ghcr.io/<owner>/yolozu:0.1.0 doctor --output -
docker run --rm ghcr.io/<owner>/yolozu-demo:0.1.0 demo continual --method ewc_replay
```

Publish trigger:
- Push a tag `vX.Y.Z` to run `.github/workflows/container.yml`.
- If the tag existed before the workflow was added, run it manually via GitHub Actions (workflow_dispatch) or cut a new tag.

Details: [deploy/docker/README.md](deploy/docker/README.md)

### GPU notes
- GPU is supported (training/inference): install CUDA-enabled PyTorch in your environment and use `--device cuda:0`.
- CI/dev does not require GPU; many checks are CPU-friendly.

## Training scaffold (RT-DETR pose)

The trainer implementation lives in `rtdetr_pose/rtdetr_pose/train_minimal.py` (source-checkout wrapper: `rtdetr_pose/tools/train_minimal.py`).

Quickest path (source checkout):

```bash
python3 -m pip install -r requirements-test.txt
bash tools/fetch_coco128.sh
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --config rtdetr_pose/configs/base.json \
  --max-steps 50 \
  --run-dir runs/train_minimal_smoke
```

Config-driven path (recommended for repeatability):

```bash
yolozu train configs/examples/train_setting.yaml
```

For quick experiments, set `--run-dir`, which writes a standard artifact set:
- `metrics.jsonl` (+ final `metrics.json` / `metrics.csv`)
- `checkpoint.pt` (+ optional `checkpoint_bundle.pt`)
- `model.onnx` (+ `model.onnx.meta.json`)
- `run_record.json` (git SHA / platform / args)

For production-style repeatability (fixed paths, resume, best/last, parity gate), use the **run contract**:

```bash
yolozu train configs/examples/train_contract.yaml --run-id exp01

# Resume (full state: model/optim/sched/amp/ema/step + RNG)
yolozu train configs/examples/train_contract.yaml --run-id exp01 --resume

# Smoke wiring check (1 step then exit 0)
yolozu train configs/examples/train_contract.yaml --run-id exp01 --dry-run
```

Contracted artifacts (fixed paths):
- `runs/<run_id>/checkpoints/{last,best}.pt`
- `runs/<run_id>/reports/train_metrics.jsonl` (1 line per optimizer step)
- `runs/<run_id>/reports/val_metrics.jsonl`
- `runs/<run_id>/reports/config_resolved.yaml`
- `runs/<run_id>/reports/run_meta.json`
- `runs/<run_id>/reports/onnx_parity.json` (Torch vs ONNXRuntime; fails run on drift by default)
- `runs/<run_id>/exports/model.onnx` (+ `model.onnx.meta.json`)

Best definition: max `map50_95` on validation.

Run contract spec: [`docs/run_contract.md`](docs/run_contract.md)

Trainer core features (implemented):
- Full resume (model/optim/sched/AMP scaler/EMA/progress + RNG) via `--resume-from` or contracted `--resume`.
- NaN/Inf guard with skip + LR decay knobs (`--stop-on-non-finite-loss`, `--non-finite-max-skips`, `--non-finite-lr-decay`).
- Grad clipping (`--clip-grad-norm`, recommended >0 for pose/TTT/MIM stability).
- AMP (`--amp {none,fp16,bf16}`), EMA (`--use-ema` + `--ema-eval`), DDP (`--ddp` via `torchrun`).
- (Optional) `torch.compile`: `--torch-compile` (+ `--torch-compile-*`; experimental; falls back by default if compile fails).
- (Optional) torchao quantization / QLoRA: `--torchao-quant {int8wo,int4wo}` / `--qlora` (experimental; requires torchao).
- Lightweight aug: multiscale (`--multiscale`), hflip (`--hflip-prob`), photometric HSV/grayscale/noise/blur (`--hsv-*`, `--gray-prob`, `--gaussian-noise-*`, `--blur-*`; effective when `--real-images` is used).
- Validation cadence: epoch (`--val-every`) and step-based (`--val-every-steps`).
- Early stop: `--early-stop-patience` (+ `--early-stop-min-delta`).

Common next checks:

```bash
python3 tools/plot_metrics.py --jsonl runs/train_minimal_smoke/metrics.jsonl --out reports/train_loss.png
python3 tools/export_predictions.py --adapter rtdetr_pose --config rtdetr_pose/configs/base.json --checkpoint runs/train_minimal_smoke/checkpoint.pt --max-images 20 --wrap --output reports/predictions.json
python3 tools/eval_coco.py --dataset data/coco128 --predictions reports/predictions.json --bbox-format cxcywh_norm --max-images 20 --dry-run
```

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

### Compatibility (YOLOv8 / YOLO11 / YOLOX)

- Ultralytics YOLOv8 / YOLO11: if your dataset root contains `images/train` + `labels/train` (and `images/val` + `labels/val`),
  it is already compatible. Use `yolozu validate dataset /path/to/dataset --strict`.
  - You can also pass an Ultralytics `data.yaml` as `--dataset` (expects `path:` + `train:`/`val:` pointing to `images/<split>`).
- YOLOX: common setups use COCO JSON (`instances_*.json`). Convert once with `tools/prepare_coco_yolo.py`
  to generate YOLO-format labels (and an optional `dataset.json` descriptor) under a YOLOZU-compatible dataset root.
  - If you want a **read-only** wrapper (no label txt generation), use import adapters: [`docs/import_adapters.md`](docs/import_adapters.md).

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

## Symmetry-aware utilities (object symmetry checks)

- Symmetry specs live in `configs/runtime/symmetry.json` (validated loader: `yolozu.config.load_symmetry_map`).
- Core ops: `yolozu/symmetry.py` (types: `none`, `Cn`/`C2`/`C4`, `Cinf`).
- Symmetry-aware template verification: `yolozu/template_verification.py`.
- Tests: `python3 -m pytest -q tests/test_symmetry.py tests/test_template_verification.py`.

## Evaluation / contracts (stable)

This repo evaluates models through a stable predictions JSON format:
- Schema doc: [docs/predictions_schema.md](docs/predictions_schema.md)
- Machine-readable schema: [schemas/predictions.schema.json](schemas/predictions.schema.json)

Adapters power `tools/export_predictions.py --adapter <name>` and follow:
- [docs/adapter_contract.md](docs/adapter_contract.md)

## Export + evaluate predictions (TTT supported)

There are two common workflows:

### A) Evaluate precomputed `predictions.json` (no torch required)

If you run real inference elsewhere (PyTorch/ONNXRuntime/TensorRT/etc.), you can evaluate this repo without installing heavy deps locally.

- Validate the JSON:
  - `python3 tools/validate_predictions.py reports/predictions.json`
- Consume predictions locally:
  - `yolozu test configs/examples/test_setting.yaml --adapter precomputed --predictions reports/predictions.json --max-images 50`
  - `python3 tools/run_scenarios.py --adapter precomputed --predictions reports/predictions.json --max-images 50`

### B) Export predictions in this repo (Torch/ONNXRuntime/TensorRT)

- Torch backend (`rtdetr_pose`, supports **TTA + TTT**):
  - Baseline: `python3 tools/yolozu.py export --backend torch --checkpoint /path/to.ckpt --device cuda --max-images 50 --output reports/predictions.json`
  - TTT (Tent, safe preset): `python3 tools/yolozu.py export --backend torch --checkpoint /path/to.ckpt --device cuda --max-images 50 --ttt --ttt-preset safe --ttt-reset sample --ttt-log-out reports/ttt_log_safe.json --output reports/predictions_ttt_safe.json`
  - TTT batch/chunk knobs: add `--ttt-batch-size <N>` and `--ttt-max-batches <K>` to cap adaptation cost (example: `--ttt-batch-size 4 --ttt-max-batches 8`).
  - TTT reset behavior: use `--ttt-reset stream` for one adaptation phase then fast prediction, or `--ttt-reset sample` for per-image/per-batch reset-ablation runs.
  - Note: `tools/yolozu.py export` always writes the wrapped `{ "predictions": [...] }` form (so `--wrap` is not needed).
  - Note: TTT is supported in the repo tooling (`python3 tools/yolozu.py ...`) on the torch backend; the pip CLI `yolozu export`
    is intentionally smoke-only (dummy/labels).
  - Recommended protocol + rationale (domain shift, presets, guards): [docs/ttt_protocol.md](docs/ttt_protocol.md)
- ONNXRuntime/TensorRT backends: use `python3 tools/yolozu.py export --backend onnxrt|trt ...` (TTT is torch-only; use TTA or export precomputed predictions for other backends).

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
