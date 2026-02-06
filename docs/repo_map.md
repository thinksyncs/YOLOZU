# Repo Map

This workspace currently contains scaffolding utilities and tests. No full training/inference pipeline is present yet.

## Key paths
- `yolozu/`: symmetry + commonsense utilities (math, constraints, gates, geometry, jitter, scenario suite)
- `tests/`: unit/integration tests for the utilities
- `tools/`: small scripts (ablation, baseline, benchmark, smoke run)
- `data/coco128/`: tiny COCO dataset for smoke tests
- `tools/manifest.json`: machine-readable registry of CLI entrypoints (see `docs/tools_index.md`)
