# Sweep Examples: TTT, Threshold, and Gate Weight Tuning

This document provides example sweep configurations for systematically exploring:
1. **TTT (Test-Time Training) parameters** (method, steps, learning rate, reset policy)
2. **Score thresholds** (detection confidence filtering)
3. **Gate weights** (score fusion for detection/template/uncertainty)

All sweeps use the `hpo_sweep.py` harness (or the `yolozu.py sweep` wrapper).

## Overview

The sweep harness executes parameterized commands, collects metrics, and writes CSV/Markdown tables.
Each sweep config is a JSON file with:
- `base_cmd`: command template with `{param}` placeholders for swept params and `$ENV_VAR` for fixed settings
- `param_grid`: dictionary of parameter names → list of values
- `env`: environment variables for fixed settings (dataset path, checkpoint, etc.)
- `metrics.path`: where to find the output metrics JSON
- `metrics.keys`: which metrics to extract (optional; if empty, stores entire JSON)

**Note**: Fixed settings (dataset, checkpoint, device) should be set as environment variables in the `env` section,
while swept parameters use `{param}` placeholders in `base_cmd`.

## 1. TTT Parameter Sweep

**Purpose**: Find optimal TTT hyperparameters (method, steps, lr, reset policy) for a given checkpoint and dataset.

**Example config**: [`docs/sweep_ttt_example.json`](sweep_ttt_example.json)

### Parameters swept

- `ttt_method`: `["tent", "mim"]` — TTT algorithm (Tent or MIM)
- `ttt_steps`: `[1, 3, 5, 10]` — Number of adaptation steps per sample/stream
- `ttt_lr`: `[1e-5, 5e-5, 1e-4, 5e-4]` — Learning rate
- `ttt_reset`: `["sample", "stream"]` — Reset policy (per-sample or stream-level)

**Total runs**: 2 × 4 × 4 × 2 = 64 configurations

### Usage

```bash
# Prepare a fixed eval subset for reproducibility
python3 tools/make_subset_dataset.py \
  --dataset data/coco128 \
  --split train2017 \
  --n 50 \
  --seed 0 \
  --out reports/coco128_50

# Edit sweep_ttt_example.json to update env vars for your setup:
# - DATASET: path to dataset
# - CHECKPOINT: path to checkpoint
# - DEVICE: cuda:0 or cpu

# Then run the sweep
python3 tools/yolozu.py sweep --config docs/sweep_ttt_example.json --resume

# Or directly with hpo_sweep.py
python3 tools/hpo_sweep.py --config docs/sweep_ttt_example.json --resume
```

**Outputs**:
- `reports/sweep_ttt.jsonl` — one line per run with params + metrics
- `reports/sweep_ttt.csv` — tabular format for plotting
- `reports/sweep_ttt.md` — Markdown table for quick review

### Evaluation

After running the sweep, evaluate each prediction file to get mAP scores:

```bash
# Example: evaluate one run
python3 tools/eval_coco.py \
  --dataset reports/coco128_50 \
  --split train2017 \
  --predictions runs/sweep_ttt/tent-sample-steps-5-lr-0.0001/predictions.json \
  --bbox-format cxcywh_norm
```

Or batch-evaluate all runs and merge metrics back into the sweep results (custom script recommended).

### Notes

- **Baseline**: Run the same command with `--no-ttt` (or remove `--ttt`) for a zero-TTT baseline.
- **Domain shift**: TTT is most effective when there's a domain gap (e.g., COCO → BDD100K, or corrupted images).
- **Reproducibility**: Use `--ttt-seed <int>` for fixed randomness in masking/augmentations.

---

## 2. Score Threshold Sweep

**Purpose**: Tune the detection confidence threshold to maximize mAP or other metrics.

**Example config**: [`docs/sweep_threshold_example.json`](sweep_threshold_example.json)

### Parameters swept

- `score_threshold`: `[0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]`

**Total runs**: 8 configurations

### Usage

```bash
# Run sweep
python3 tools/yolozu.py sweep --config docs/sweep_threshold_example.json --resume
```

**Command breakdown**:
1. Export predictions with varying thresholds
2. Evaluate each with COCO mAP (`eval_coco.py`)
3. Extract `metrics.map50`, `metrics.map50_95`, `metrics.ar100` from the metrics JSON (note: `eval_coco.py` outputs `metrics.ar100`, not `mar_100`)

**Outputs**:
- `reports/sweep_threshold.jsonl`
- `reports/sweep_threshold.csv`
- `reports/sweep_threshold.md`

### Analyzing results

Open `reports/sweep_threshold.csv` and plot `score_threshold` vs `map50_95` to find the optimal threshold.

Example (requires pandas/matplotlib):

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("reports/sweep_threshold.csv")
df = df.sort_values("params.score_threshold")

plt.plot(df["params.score_threshold"], df["metrics.map50_95"], marker="o")
plt.xlabel("Score Threshold")
plt.ylabel("mAP@50-95")
plt.title("Threshold vs mAP")
plt.grid(True)
plt.savefig("reports/threshold_sweep.png")
```

---

## 3. Gate Weight Sweep

**Purpose**: Tune inference-time gate weights for score fusion (detection + template + uncertainty).

**Example config**: [`docs/sweep_gate_weights_example.json`](sweep_gate_weights_example.json)

### Background

YOLOZU supports lightweight inference-time rescoring:
```
final_score = w_det * score_det + w_tmp * score_tmp - w_unc * (sigma_z + sigma_rot)
```

The `tune_gate_weights.py` tool performs grid search over these weights **offline on CPU** (no retraining required).

### Parameters swept

- `grid_det`: `["1.0"]` — keep detection weight fixed at 1.0
- `grid_tmp`: `["0.0,0.25,0.5,0.75,1.0", "0.0,0.5,1.0"]` — template score weight
- `grid_unc`: `["0.0,0.25,0.5,0.75,1.0", "0.0,0.5,1.0"]` — uncertainty penalty weight
- `metric`: `["map50_95", "map50"]` — optimization target

**Total runs**: 1 × 2 × 2 × 2 = 8 configurations (each performs its own inner grid search)

### Usage

```bash
# First, generate predictions with uncertainty estimates
python3 tools/export_predictions.py \
  --adapter rtdetr_pose \
  --dataset data/coco128 \
  --split train2017 \
  --checkpoint runs/rtdetr_pose/checkpoint.pt \
  --wrap \
  --output reports/predictions_rtdetr_pose.json

# Run gate weight sweep
python3 tools/yolozu.py sweep --config docs/sweep_gate_weights_example.json --resume
```

**Outputs**:
- `reports/sweep_gate_weights.jsonl`
- `reports/sweep_gate_weights.csv`
- `reports/sweep_gate_weights.md`

Each run produces a `gate_tuning_report.json` metrics report with:
- `metrics.tuning.best.det`, `metrics.tuning.best.tmp`, `metrics.tuning.best.unc`: optimal gate weights found
- `metrics.tuning.best.map50`, `metrics.tuning.best.map50_95`: mAP scores achieved with those weights
- additional tuning rows under `metrics.tuning` that the sweep harness can aggregate into CSV/Markdown

### Notes

- **No GPU required**: Gate tuning runs on CPU using `simple_map` proxy.
- **Uncertainty fields**: Requires predictions with `sigma_z`, `sigma_rot` (RTDETRPose with `use_uncertainty=true`).
- **Template scores**: Optionally add `score_tmp_sym` per detection (from external template matcher).

---

## 4. Combined Sweeps

You can nest sweeps or chain them:

### Example: TTT + Threshold sweep

1. Run TTT sweep to find best TTT config
2. Pick the best TTT config from step 1
3. Run threshold sweep with that TTT config

Or do a Cartesian product (TTT params × thresholds) — note this can be large!

---

## Advanced: Custom metrics extraction

If your command writes a custom JSON structure, adjust `metrics.keys` to extract the right fields:

```json
{
  "metrics": {
    "path": "{run_dir}/custom_metrics.json",
    "keys": ["model.map50_95", "timing.inference_ms", "meta.git_sha"]
  }
}
```

The harness uses dot-notation to traverse nested dicts.

---

## Tips

1. **Use `--resume`**: Skip already-completed runs (based on `run_id` in results JSONL).
2. **Use `--max-runs N`**: Cap the number of runs for quick tests.
3. **Use `--dry-run`**: Print commands without executing (useful for debugging config).
4. **Pin dataset**: Use `make_subset_dataset.py` for reproducible evaluation subsets.
5. **Multiple seeds**: For stochastic methods (TTT, TTA), run sweeps with different seeds and aggregate results.

---

## Summary Table

| Sweep Type | Config File | Typical Runs | Outputs | Use Case |
|------------|-------------|--------------|---------|----------|
| TTT | `sweep_ttt_example.json` | 64 | `sweep_ttt.{jsonl,csv,md}` | Find best TTT hyperparams |
| Threshold | `sweep_threshold_example.json` | 8 | `sweep_threshold.{jsonl,csv,md}` | Find optimal score cutoff |
| Gate Weights | `sweep_gate_weights_example.json` | 8 | `sweep_gate_weights.{jsonl,csv,md}` | Tune inference-time score fusion |

All sweeps produce **CSV/MD tables** for easy plotting and comparison.

---

## References

- Sweep harness: [`tools/hpo_sweep.py`](../tools/hpo_sweep.py)
- TTT protocol: [`docs/ttt_protocol.md`](ttt_protocol.md)
- Gate weight tuning: [`docs/gate_weight_tuning.md`](gate_weight_tuning.md)
- Unified CLI: [`tools/yolozu.py`](../tools/yolozu.py)
