# Run Contract (production-style training artifacts)

This repo’s minimal trainer (`rtdetr_pose/tools/train_minimal.py`) supports a **run contract** mode that pins:
- reproducibility inputs (config + seed + environment metadata)
- fixed output paths (checkpoints, metrics JSONL, exports)
- safety gates (full resume, non-finite guards, parity checks)

The goal is: **the same YAML can always do** `train → resume → export → eval`, and artifacts are always in the same place.

## Enable

Enable the contract by passing either:
- `--run-contract`, or
- `--run-id <id>` (implicitly enables the contract)

Recommended (repo checkout):

```bash
yolozu dev train configs/examples/train_contract.yaml --run-id exp01
```

## Required inputs

When the run contract is enabled:
- `--config <train_setting.yaml>` is required (this is the trainer settings file)
- `config_version: 1` is required (set in the YAML; enforced at startup)

Recommended to set explicitly in the YAML/config:
- `dataset_root`
- `seed`
- `device` (`cpu` / `cuda:0`, ...)
- `amp` / precision (`none` / `fp16` / `bf16`)
- distributed settings (`ddp: true` when using `torchrun`)

## Outputs (fixed paths)

All outputs live under:

```
runs/<run_id>/
```

Required artifacts:
- `runs/<run_id>/checkpoints/last.pt`
- `runs/<run_id>/checkpoints/best.pt`
- `runs/<run_id>/reports/train_metrics.jsonl` (1 line per optimizer step)
- `runs/<run_id>/reports/val_metrics.jsonl`
- `runs/<run_id>/reports/config_resolved.yaml` (fully-resolved args)
- `runs/<run_id>/reports/run_meta.json` (git SHA, torch/cuda, host, cmdline, etc.)

Export + verification artifacts (enabled by default):
- `runs/<run_id>/exports/model.onnx`
- `runs/<run_id>/exports/model.onnx.meta.json`
- `runs/<run_id>/reports/onnx_parity.json` (Torch vs ONNXRuntime diff stats)

### Best checkpoint definition

`best.pt` is updated when validation `map50_95` improves (maximization).

## Resume (full state)

Contract resume is:

```bash
yolozu dev train configs/examples/train_contract.yaml --run-id exp01 --resume
```

This resumes from:
- `runs/<run_id>/checkpoints/last.pt`

The contracted checkpoint bundle restores (when present):
- model weights
- optimizer + scheduler state
- AMP scaler state
- EMA state
- progress counters (epoch/global_step)
- RNG state (torch/cuda + python + optional numpy)

## ONNX parity gate

When the contract is enabled, parity runs by default and:
- writes stats to `runs/<run_id>/reports/onnx_parity.json`
- fails the run if drift exceeds thresholds (`--parity-policy fail`)

You can relax this behavior with:
- `--parity-policy warn`

Dependencies:
- ONNX export requires `onnx`
- parity requires `onnxruntime`
  - easiest: `python3 -m pip install 'yolozu[onnxrt]'`

## DDP (1-node multi-GPU)

Use `torchrun` and `--ddp`:

```bash
torchrun --nproc_per_node=2 rtdetr_pose/tools/train_minimal.py \
  --config configs/examples/train_contract.yaml \
  --run-id exp01 \
  --ddp
```

Only rank 0 writes contracted artifacts; non-zero ranks run training and synchronize at barriers.

