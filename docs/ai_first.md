# AI-first usage guide

This page is for agentic workflows (Codex/CLI agents, scripting copilots, CI bots).
Goal: make safe, deterministic automation easy.

## Stable entry points

- Unified command surface: `python3 tools/yolozu.py --help`
- Machine-readable tool registry: `tools/manifest.json`
- Contract schemas: `docs/schemas/*.schema.json`
- Predictions contract docs: `docs/predictions_schema.md`
- Adapter contract docs: `docs/adapter_contract.md`

## Recommended automation flow

1) **Validate inputs/contracts**
```bash
python3 tools/validate_dataset.py --dataset /path/to/yolo --strict
python3 tools/validate_predictions.py /path/to/predictions.json --strict
```

2) **Run protocol-pinned evaluation**
```bash
python3 tools/eval_suite.py \
  --protocol yolo26 \
  --dataset /path/to/yolo \
  --predictions-glob '/path/to/pred_*.json' \
  --output reports/eval_suite.json
```

3) **Store report artifacts**
- JSON output path from stdout
- Optional HTML report paths when a tool supports `--html`

## Path rules (for robust agents)

- CLI relative input paths: resolve from current working directory.
- Config-file relative paths (where supported, e.g. `tools/tune_gate_weights.py`):
  resolve from config-file directory.
- Relative outputs: write under current working directory.
- Input compatibility fallback: repo-root fallback remains enabled for legacy scripts.

## Safety checks before write actions

- Dry runs where available (`--dry-run`)
- Validate caps/ranges (`--max-images`, thresholds)
- Keep outputs under `reports/` or explicit run dirs

## Quality gate command (recommended handoff)

```bash
./.venv/bin/ruff check .
python3 -m unittest -q
```
