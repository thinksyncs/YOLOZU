# Continual learning (anti-forgetting) for `rtdetr_pose`

This repo supports **continual fine-tuning** on multiple datasets/tasks while mitigating catastrophic forgetting.

The baseline strategy is:

- **Memoryless**: self-distillation from the previous checkpoint (`--self-distill-from`) so the new model stays close to the old one (SDFT-style objective; reverse-KL on logits by default).
- **Memory**: add a small **replay buffer** (default 50 images, reservoir sampling) and train on *(current task + replay)* while also self-distilling.
- Optional: **LoRA** to restrict trainable parameters (parameter-efficient continual fine-tuning).

## Quick start (domain-incremental)

1) Create a continual config (start from the example):

- `configs/continual/rtdetr_pose_domain_inc_example.yaml`

2) Run continual fine-tuning:

```bash
python3 rtdetr_pose/tools/train_continual.py \
  --config configs/continual/rtdetr_pose_domain_inc_example.yaml
```

To run **memoryless**, set `continual.replay_size: 0` in the config (or pass `--replay-size 0`).

3) Evaluate forgetting (mAP proxy + CL summary metrics):

```bash
python3 tools/eval_continual.py \
  --run-json runs/continual/<run>/continual_run.json \
  --device cpu \
  --max-images 50
```

This writes:
- `runs/continual/<run>/continual_eval.json`
- `runs/continual/<run>/continual_eval.html`

## Outputs (train)

`train_continual.py` creates a run folder under `runs/continual/` and writes:

- `continual_run.json` (tasks, checkpoints, config hash, run record)
- `replay_buffer.json` (buffer summary)
- Per-task folders:
  - `checkpoint.pt` (weights-only)
  - `checkpoint_bundle.pt` (optional; if enabled in `train_minimal.py`)
  - `metrics.jsonl/json/csv`
  - `run_record.json`

## Notes / caveats

- The current continual evaluation uses `yolozu.simple_map` (CPU-friendly proxy). For full COCO mAP you can switch your workflow to `tools/eval_coco.py` with `pycocotools` installed.
- `rtdetr_pose` dataset loading currently scans `*.jpg` files under `images/<split>/` (extend if you use png).
