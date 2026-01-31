# Hyperparameter sweep harness

This repo does not include a full training loop yet, but you can still run
parameter sweeps against **any** external training command. The sweep harness
executes commands, collects a metrics JSON, and writes **CSV/MD tables**.

## Quick start (mock run)

```bash
python3 tools/hpo_sweep.py --config docs/hpo_sweep_example.json --resume
```

Outputs:
- `reports/hpo_sweep.jsonl`
- `reports/hpo_sweep.csv`
- `reports/hpo_sweep.md`

## Config fields (JSON)

Required:
- `base_cmd`: command template (uses `{param}` placeholders)
- `param_grid` or `param_list`

Optional:
- `run_dir`: output directory template (default `runs/hpo/{run_id}`)
- `metrics.path`: JSON path template to read metrics
- `metrics.keys`: list of key paths to extract from metrics JSON
- `result_jsonl`, `result_csv`, `result_md`
- `env`: extra environment variables
- `shell`: run command via shell (default true)

## Notes
- Use `--resume` to skip runs already present in the results JSONL.
- The harness does not assume any specific training framework.
- If `metrics.keys` is empty, the whole JSON object is stored in the results.
