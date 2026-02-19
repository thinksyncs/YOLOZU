# SAR robustness impact benchmark (phase 1)

This protocol quantifies SAR robustness gains and side effects against CoTTA/EATA references.

## Goal

- quantify robustness gain in final-loss and guard behavior,
- quantify side effects in overhead and variance,
- produce a reproducible go/no-go decision artifact for rollout.

## Inputs

Use wrapped predictions outputs from `tools/export_predictions.py` so `meta.ttt.report` is present.

- CoTTA run output (`--ttt-method cotta`)
- EATA run output (`--ttt-method eata`)
- SAR run output (`--ttt-method sar`)

## Run benchmark tool

```bash
python3 tools/benchmark_sar_robustness.py \
  --cotta reports/preds_cotta_ttt.json \
  --eata reports/preds_eata_ttt.json \
  --sar reports/preds_sar_ttt.json \
  --output-json reports/sar_robustness_report.json \
  --output-md reports/sar_robustness_report.md \
  --max-overhead-ratio 1.5 \
  --max-loss-ratio 1.05 \
  --max-variance-ratio 1.2
```

Artifacts:

- JSON: `reports/sar_robustness_report.json`
- Markdown: `reports/sar_robustness_report.md`

## Decision fields

`checks` contains:

- `robustness_gain`
- `acceptable_overhead`
- `acceptable_loss`
- `acceptable_variance`

`side_effects` contains:

- `overhead_vs_best_baseline`
- `loss_delta_vs_best_baseline`
- `variance_delta_vs_best_baseline`
- `guard_breach_delta_vs_best_baseline`

`decision` contains:

- `go` (boolean)
- `summary` (`go` or `no-go`)
- `recommended_next`

## Smoke evidence in-repo

- JSON artifact: `docs/assets/sar_robustness_smoke.json`
- Markdown summary: `docs/assets/sar_robustness_smoke.md`
