# Repo Map

This workspace contains a **contract-first evaluation harness** plus a **minimal training scaffold** (RT-DETR pose).

Two usage modes:
- **pip users**: `pip install yolozu` → stable, CPU-friendly CLI (`yolozu doctor|export|validate|eval-instance-seg|resources|demo`)
- **repo users**: source checkout unlocks additional tools (`tools/`, `rtdetr_pose/`, scenario suite runners, TensorRT pipeline helpers)

## Key paths
- `yolozu/`: pip-installable package (CLI + schemas + demos)
- `docs/`: user-facing docs (protocols, pipelines, recipes)
- `tests/`: unit/integration tests (CPU-friendly by default; GPU optional)
- `tools/`: repo-only scripts (exporters, suites, benchmarks, smoke runs)
- `rtdetr_pose/`: RT-DETR pose scaffold (training/inference/export helpers)
- `data/coco128/`: tiny COCO dataset for smoke tests (downloaded; not committed)
