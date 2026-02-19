# CoTTA drift suppression validation (phase 1)

This protocol validates phase-1 CoTTA behavior against a baseline TTT run and emits reproducible evidence artifacts.

## Goal

- Show stabilization effect versus baseline in a reproducible report.
- Confirm no unsafe parameter drift under configured safety threshold.

## Inputs

Use wrapped predictions outputs from `tools/export_predictions.py` so `meta.ttt.report` is present.

- Baseline run output (`--ttt-method tent` or existing baseline method)
- CoTTA run output (`--ttt-method cotta`)

## Generate validation report

```bash
python3 tools/eval_cotta_drift.py \
  --baseline reports/preds_baseline_ttt.json \
  --cotta reports/preds_cotta_ttt.json \
  --output-json reports/cotta_drift_report.json \
  --output-md reports/cotta_drift_report.md \
  --stability-loss-ratio-threshold 1.0 \
  --max-safe-total-update-norm 5.0
```

Artifacts:

- JSON: `reports/cotta_drift_report.json`
- Markdown: `reports/cotta_drift_report.md`

## Decision fields

`decision.stabilization_pass` is true only when all are true:

- CoTTA final-loss summary is not worse than baseline under ratio threshold.
- CoTTA guard breach count is not worse than baseline.
- Unsafe drift is not detected (`cotta.max_total_update_norm <= max_safe_total_update_norm`).

Use the JSON artifact as the reproducible evidence record (inputs + SHA256 + thresholds + decision).

## Smoke evidence in-repo

- JSON artifact: `docs/assets/cotta_drift_smoke.json`
- Markdown summary: `docs/assets/cotta_drift_smoke.md`
