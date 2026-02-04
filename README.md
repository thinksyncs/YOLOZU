# YOLOZU

Real-time monocular RGB pipeline for object detection, depth, and 6DoF pose estimation (RT-DETR-based).

- Spec: [rt_detr_6dof_geom_mim_spec_en_v0_4.md](rt_detr_6dof_geom_mim_spec_en_v0_4.md)
- Notes/TODOs: [todo_symmetry_commonsense_realtime.md](todo_symmetry_commonsense_realtime.md)
- License policy: [docs/license_policy.md](docs/license_policy.md)

---

## Copy-light: toppymicroservices

Use these as lightweight landing-page/app-store/one-pager blurbs (edit freely).


`toppymicroservices` is a pragmatic microservices stack for shipping AI-backed APIs reliably.


---

## Testing (tiny COCO)
- Install deps (CPU PyTorch): `python3 -m pip install -r requirements-test.txt`
- Fetch dataset (once): `bash tools/fetch_coco128.sh`
- Dataset: `data/coco128` (YOLO-format COCO subset, fetched from official COCO hosting).
- Smoke test: `python3 -m unittest tests/test_coco128_smoke.py`
- Core checks: `python3 -m unittest tests/test_config_loader.py tests/test_symmetry.py tests/test_metrics.py tests/test_gates_constraints.py tests/test_geometry_pipeline.py tests/test_jitter.py tests/test_scenario_suite.py tests/test_benchmark.py`
- Dataset manifest: `python3 tools/build_manifest.py`
- Adapter run: `python3 tools/run_scenarios.py`

### Notes
- GPU is not required for development/tests; CPU-only PyTorch is supported.
- If you need CUDA, install PyTorch separately for your GPU and then run: `python3 -m pip install -r requirements.txt`
- Dev extras: `python3 -m pip install -r requirements-dev.txt`

## Precomputed predictions workflow (no torch required)

If you run real inference elsewhere (PyTorch/TensorRT/etc.), you can evaluate this repo without installing heavy deps locally.

- Export predictions (in an environment where the adapter can run):
  - `python3 tools/export_predictions.py --adapter rtdetr_pose --checkpoint /path/to.ckpt --max-images 50 --wrap --output reports/predictions.json`
- Validate the JSON:
  - `python3 tools/validate_predictions.py reports/predictions.json`
- Consume predictions locally:
  - `python3 tools/run_scenarios.py --adapter precomputed --predictions reports/predictions.json --max-images 50`

Supported predictions JSON shapes:
- `[{"image": "...", "detections": [...]}, ...]`
- `{ "predictions": [ ... ] }`
- `{ "000000000009.jpg": [...], "/abs/path.jpg": [...] }` (image -> detections)

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

## Hyperparameter sweep harness

Run a configurable sweep and emit CSV/MD tables:
- `docs/hpo_sweep.md`

## Latency/FPS benchmark harness

Report latency/FPS per YOLO26 bucket and archive runs over time:
- `docs/benchmark_latency.md`

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
