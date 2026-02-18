# AI-first usage guide

This page is for agentic workflows (Codex/CLI agents, scripting copilots, CI bots).
Goal: make safe, deterministic automation easy.

## Stable entry points

- Unified command surface: `python3 tools/yolozu.py --help`
- AI-first tool registry surface: `python3 tools/yolozu.py registry --help`
- Machine-readable tool registry: `tools/manifest.json`
- Contract schemas: `docs/schemas/*.schema.json`
- Predictions contract docs: `docs/predictions_schema.md`
- Adapter contract docs: `docs/adapter_contract.md`

The `registry` subcommands are the recommended interface for agents because they provide:

- Stable discovery (`registry list --json`, `registry show --json`)
- Built-in safety gates for side effects (`registry run ...`)
- Best-effort post-run contract validation (when `contract_outputs` is declared in the manifest)

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

For agent automation, prefer running tools through the registry runner:

```bash
# Machine-readable registry discovery (stable JSON)
python3 tools/yolozu.py registry list --json > reports/tool_registry.json

# Safe execution with explicit side-effect allowlists
python3 tools/yolozu.py registry run normalize_predictions -- \
  --input reports/predictions.json \
  --output reports/predictions_norm.json
```

For tools that write outside `reports/` (for example dataset preparation under `data/`), add an explicit allowlist:

```bash
python3 tools/yolozu.py registry run --allow-write-root data prepare_coco_yolo -- \
  --coco-root /path/to/coco \
  --out data/coco-yolo
```

`registry run` blocks common footguns by default:

- tools that require network unless `--allow-network`
- tools that require GPU unless `--allow-gpu`
- writes outside allowlisted roots (default: `reports/`) unless `--allow-write-root <root>`
- absolute paths / `..` segments unless `--allow-unsafe-paths`

## Quality gate command (recommended handoff)

```bash
./.venv/bin/ruff check .
python3 -m unittest -q
```
