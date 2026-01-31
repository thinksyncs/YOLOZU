# Task aligner (multi-task loss alignment)

This repo supports a minimal **multi-task loss aligner** for RTDETRPose training via an uncertainty-weighted loss formulation.

## Uncertainty aligner

When enabled, `loss_z` and `loss_rot` are replaced by:

`exp(-s) * loss + s`

where `s` is a learned per-task log-variance proxy (`log_sigma_z`, `log_sigma_rot`).

## How to use (train_minimal)

Enable the uncertainty heads and the aligner:

```bash
python3 rtdetr_pose/tools/train_minimal.py \
  --use-matcher \
  --use-uncertainty \
  --task-aligner uncertainty
```

Loss keys emitted by `Losses(..., task_aligner="uncertainty")`:
- `loss_z_aligned` (when `log_sigma_z` is present)
- `loss_rot_aligned` (when `log_sigma_rot` is present)

