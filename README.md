# YOLOZU

Real-time monocular RGB pipeline for object detection, depth, and 6DoF pose estimation (RT-DETR-based).

- Spec: [rt_detr_6dof_geom_mim_spec_en_v0_4.md](rt_detr_6dof_geom_mim_spec_en_v0_4.md)
- Notes/TODOs: [todo_symmetry_commonsense_realtime.md](todo_symmetry_commonsense_realtime.md)

---

## Copy-light: toppymicroservices

Use these as lightweight landing-page/app-store/one-pager blurbs (edit freely).

### Option A (one-liner)
`toppymicroservices` is a pragmatic microservices stack for shipping AI-backed APIs reliably.

### Option B (short paragraph)
`toppymicroservices` helps you deploy and operate small, well-defined services with clean boundaries—so you can iterate fast without losing observability, reliability, or control.

### Option C (bullets)
- Build small services with clear contracts
- Ship safely with simple rollout patterns
- Keep operations predictable with logging/metrics first
- Scale the pieces that actually need scaling

---

## Testing (tiny COCO)
- Install deps (CPU PyTorch): `python3 -m pip install -r requirements-test.txt`
- Fetch dataset (once): `bash tools/fetch_coco128.sh`
- Dataset: `data/coco128` (YOLO-format COCO subset).
- Smoke test: `python3 -m unittest tests/test_coco128_smoke.py`
- Core checks: `python3 -m unittest tests/test_config_loader.py tests/test_symmetry.py tests/test_metrics.py tests/test_gates_constraints.py tests/test_geometry_pipeline.py tests/test_jitter.py tests/test_scenario_suite.py tests/test_benchmark.py`
- Dataset manifest: `python3 tools/build_manifest.py`
- Adapter run: `python3 tools/run_scenarios.py`

### Notes
- GPU is not required for development/tests; CPU-only PyTorch is supported.
- If you need CUDA, install PyTorch separately for your GPU and then run: `python3 -m pip install -r requirements.txt`
- Dev extras: `python3 -m pip install -r requirements-dev.txt`

## Deployment notes
- Keep symmetry/commonsense logic in lightweight postprocess utilities, outside any inference graph export.
