# EATA stability/efficiency benchmark (phase 1)

This protocol measures EATA tradeoffs against baseline TTT on fixed small-batch mixed-shift slices.

## Goal

- quantify stability impact (loss/guard behavior),
- quantify efficiency impact (runtime overhead),
- derive recommended default EATA knobs for rollout.

## Inputs

Use wrapped predictions outputs from `tools/export_predictions.py` so `meta.ttt.report` is present.

- Baseline run output (`--ttt-method tent`)
- EATA run output (`--ttt-method eata`)

## Run benchmark tool

```bash
python3 tools/benchmark_eata_stability.py \
  --baseline reports/preds_baseline_ttt.json \
  --eata reports/preds_eata_ttt.json \
  --output-json reports/eata_benchmark.json \
  --output-md reports/eata_benchmark.md \
  --max-overhead-ratio 1.5 \
  --max-loss-ratio 1.05 \
  --min-selected-ratio 0.1
```

Artifacts:

- JSON: `reports/eata_benchmark.json`
- Markdown: `reports/eata_benchmark.md`

## Tradeoff fields

`tradeoff` contains:

- `loss_delta_vs_baseline`
- `guard_breach_delta`
- `overhead_ratio`
- `loss_ratio`

`checks` reports pass/fail for stability and efficiency constraints.

## Recommended defaults

`recommended_defaults` emits a conservative phase-1 configuration (`eata_safe` + threshold knobs).
Use this as the initial rollout profile when checks pass.

## Smoke evidence in-repo

- JSON artifact: `docs/assets/eata_benchmark_smoke.json`
- Markdown summary: `docs/assets/eata_benchmark_smoke.md`
