# EATA stability/efficiency benchmark

- Generated UTC: 2026-02-19T12:46:10Z
- Recommend EATA defaults: `True`

## Baseline vs EATA

| Variant | Mean Final Loss | Guard Breaches | Mean Seconds | Mean Selected Ratio |
|---|---:|---:|---:|---:|
| baseline | 1.000000 | 1 | 0.500000 | 0.000000 |
| eata | 0.900000 | 0 | 0.650000 | 0.550000 |

## Tradeoff summary

- loss_delta_vs_baseline: `-0.100000`
- guard_breach_delta: `-1`
- overhead_ratio: `1.300000`

## Recommended defaults

- enabled: `True`
- ttt_method: `eata`
- ttt_preset: `eata_safe`
- ttt_update_filter: `lora_norm_only`
- ttt_eata_conf_min: `0.2`
- ttt_eata_entropy_min: `0.05`
- ttt_eata_entropy_max: `3.0`
- ttt_eata_min_valid_dets: `1`
- ttt_eata_anchor_lambda: `0.001`
