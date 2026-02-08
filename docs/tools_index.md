# Tools index (AI-friendly)

This repo treats `tools/` as a stable, scriptable interface layer on top of the lightweight `yolozu/` core.

## Unified CLI (recommended entrypoint)

For most day-to-day flows, start with:

- `python3 tools/yolozu.py doctor ...`
- `python3 tools/yolozu.py export --backend {dummy,torch,onnxrt,trt} ...`
- `python3 tools/yolozu.py predict-images --input-dir /path/to/images ...`
- `python3 tools/yolozu.py eval-keypoints --dataset /path/to/yolo --predictions /path/to/predictions.json ...`
- `python3 tools/yolozu.py eval-instance-seg --dataset /path/to/yolo --predictions /path/to/instance_seg_predictions.json ...`
- `python3 tools/yolozu.py sweep --config docs/hpo_sweep_example.json ...`

## Dataset helpers

- `python3 tools/make_subset_dataset.py --dataset /path/to/yolo --n 500 --seed 0 --out reports/subset_dataset`

## Evaluation helpers

- Keypoints (PCK): `python3 tools/eval_keypoints.py --dataset /path/to/yolo --predictions reports/predictions.json --output reports/keypoints_eval.json`

## Machine-readable tool registry

- Tool manifest: `tools/manifest.json`
- Manifest schema: `docs/schemas/tools_manifest.schema.json`
- Validator: `python3 tools/validate_tool_manifest.py`

The manifest is intended for:
- AI agents that need to discover available CLI entrypoints + their I/O contracts
- humans who want a quick map of “what command do I run to do X?”

## Contracts (recommended)

Most flows in this repo pass data as JSON artifacts:
- `predictions_json`: per-image detections JSON (validate with `tools/validate_predictions.py`)
- `metrics_report_json`: stable report payloads (`yolozu.metrics_report.build_report`)
- JSON Schemas live under `docs/schemas/` and are referenced from `tools/manifest.json` contracts.

When adding a new tool, prefer:
1) reading inputs from file paths / flags
2) writing outputs to a deterministic path (default under `reports/`)
3) printing the output path to stdout

## Adding a new tool (checklist)

- Add the script under `tools/` (thin CLI; keep logic in `yolozu/` when possible)
- Add an entry to `tools/manifest.json` with:
  - `id`, `entrypoint`, `runner`, `summary`
  - at least one runnable `examples[].command`
  - `contracts.{consumes,produces}` when applicable
- Run: `python3 tools/validate_tool_manifest.py`
