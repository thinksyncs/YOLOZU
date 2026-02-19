# Manifest coverage audit (YOLOZU-1m2.2)

- Source of truth: `tools/manifest.json`
- Tools scanned: `52`
- Tools with gaps: `12`
- Findings total: `12`

## Findings by category

- `missing_tool_fields`: `8`
- `effects_missing_fixed_writes`: `4`

## Per-tool gap list and fix direction

### benchmark_keypoints_eval
- path: `effects.fixed_writes`
  - category: `effects_missing_fixed_writes`
  - missing: `fixed_writes`
  - fix direction: Declare effects.fixed_writes (use empty list if none).

### benchmark_latency
- path: `effects.fixed_writes`
  - category: `effects_missing_fixed_writes`
  - missing: `fixed_writes`
  - fix direction: Declare effects.fixed_writes (use empty list if none).

### build_manifest
- path: `tool`
  - category: `missing_tool_fields`
  - missing: `inputs`
  - fix direction: Add missing top-level declarative fields to each tool entry.

### build_trt_engine
- path: `tool`
  - category: `missing_tool_fields`
  - missing: `outputs`
  - fix direction: Add missing top-level declarative fields to each tool entry.

### check_license_policy
- path: `tool`
  - category: `missing_tool_fields`
  - missing: `inputs`
  - fix direction: Add missing top-level declarative fields to each tool entry.

### export_predictions_onnxrt
- path: `tool`
  - category: `missing_tool_fields`
  - missing: `outputs`
  - fix direction: Add missing top-level declarative fields to each tool entry.

### export_predictions_trt
- path: `tool`
  - category: `missing_tool_fields`
  - missing: `outputs`
  - fix direction: Add missing top-level declarative fields to each tool entry.

### export_trt
- path: `tool`
  - category: `missing_tool_fields`
  - missing: `outputs`
  - fix direction: Add missing top-level declarative fields to each tool entry.

### fetch_coco128
- path: `tool`
  - category: `missing_tool_fields`
  - missing: `inputs`
  - fix direction: Add missing top-level declarative fields to each tool entry.

### report_dependency_licenses
- path: `effects.fixed_writes`
  - category: `effects_missing_fixed_writes`
  - missing: `fixed_writes`
  - fix direction: Declare effects.fixed_writes (use empty list if none).

### run_scenarios
- path: `effects.fixed_writes`
  - category: `effects_missing_fixed_writes`
  - missing: `fixed_writes`
  - fix direction: Declare effects.fixed_writes (use empty list if none).

### yolozu
- path: `tool`
  - category: `missing_tool_fields`
  - missing: `outputs`
  - fix direction: Add missing top-level declarative fields to each tool entry.

## Next fix order (recommended)

1. Add missing top-level declarative fields (`inputs/effects/outputs/examples/platform`) for impacted tools.
2. Normalize `effects` declarations (`writes` + `fixed_writes`) so every write path is explicit.
3. Ensure each tool has at least one runnable `examples[].command` and complete input/output metadata.
4. Tighten validator enforcement in `tools/validate_tool_manifest.py` and add regression tests.

- Machine-readable artifact: `/Users/akira/YOLOZU/reports/manifest_coverage_gaps.json`
