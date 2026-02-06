# Tools index (AI-friendly)

This repo treats `tools/` as a stable, scriptable interface layer on top of the lightweight `yolozu/` core.

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
