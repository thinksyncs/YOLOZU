# Continual learning (anti-forgetting) for `rtdetr_pose`

This repo supports **continual fine-tuning** on multiple datasets/tasks while mitigating catastrophic forgetting.

The baseline strategy is:

- **Memoryless**: self-distillation from the previous checkpoint (`--self-distill-from`) so the new model stays close to the old one (SDFT-style objective; reverse-KL on logits by default).
- **Memory**: add a small **replay buffer** (default 50 images, reservoir sampling) and train on *(current task + replay)* while also self-distilling.
- Optional: **LoRA** to restrict trainable parameters (parameter-efficient continual fine-tuning).

## 0-minute start (pip users; CPU)

If you want to **see continual-learning behavior** (domain shift + forgetting mitigation) without downloading datasets:

```bash
python3 -m pip install 'yolozu[demo]'
yolozu demo continual --compare --markdown
```

This is a **toy synthetic** demo (not an image model). For real continual fine-tuning on image datasets, use the `rtdetr_pose` workflow below.

## Quick start (domain-incremental)

1) Create a continual config (start from the example):

- `configs/continual/rtdetr_pose_domain_inc_example.yaml`

2) Run continual fine-tuning:

```bash
python3 rtdetr_pose/tools/train_continual.py \
  --config configs/continual/rtdetr_pose_domain_inc_example.yaml
```

To run **memoryless**, set `continual.replay_size: 0` in the config (or pass `--replay-size 0`).

3) Evaluate forgetting (mAP proxy or pose metrics + CL summary metrics):

```bash
python3 tools/eval_continual.py \
  --run-json runs/continual/<run>/continual_run.json \
  --device cpu \
  --max-images 50
```

Pose/depth metrics (requires pose sidecar metadata in `labels/<split>/*.json`):

```bash
python3 tools/eval_continual.py \
  --run-json runs/continual/<run>/continual_run.json \
  --device cpu \
  --max-images 50 \
  --metric pose \
  --metric-key pose_success
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

## Config knobs (selected)

Under `continual:` in the YAML/JSON:

- Replay:
  - `replay_size`: buffer capacity (0 disables replay).
  - `replay_fraction`: if set, use `round(fraction * train_records)` replay samples per task instead of the full buffer.
  - `replay_per_task_cap`: optional cap of replay samples per past task (helps avoid imbalance).

- Self-distillation (memoryless baseline; enabled by default in the runner):
  - `distill.enabled`: enable/disable distillation against the previous checkpoint.
  - `distill.keys`: e.g. `"logits,bbox"`.
  - `distill.weight`, `distill.temperature`, `distill.kl`.

- DER++-style replay distillation (optional, stronger baseline):
  - `derpp.enabled`: enable DER++ loss for replay samples that have teacher outputs stored in their records.
  - `derpp.teacher_key`: record key for stored teacher outputs (default: `derpp_teacher_npz`).
  - `derpp.keys`, `derpp.weight`, `derpp.temperature`, `derpp.kl`.
  - `derpp.logits_weight`, `derpp.bbox_weight`, `derpp.other_l1_weight`.

  Notes:
  - This implementation expects **per-sample teacher outputs** to be present in the dataset records (either as an inline dict or an `.npz/.json` path).
  - It is CPU-safe (no extra teacher forward required at training time), but full validation needs real continual runs.

- Regularizers (optional; tracked per task):
  - EWC (Elastic Weight Consolidation):
    - `ewc.enabled`: enable.
    - `ewc.lambda`: penalty weight.
    - Per-task state is saved as `taskXX_*/ewc_state.pt` and used as `--ewc-state-in` for the next task.
  - SI (Synaptic Intelligence):
    - `si.enabled`: enable.
    - `si.c`: penalty weight.
    - `si.epsilon`: denominator stabilizer (default: `1e-3`).
    - Per-task state is saved as `taskXX_*/si_state.pt` and used as `--si-state-in` for the next task.

## Notes / caveats

- The current continual evaluation uses `yolozu.simple_map` (CPU-friendly proxy). For full COCO mAP you can switch your workflow to `tools/eval_coco.py` with `pycocotools` installed.
- `rtdetr_pose` dataset loading scans `*.jpg`, `*.jpeg`, and `*.png` under `images/<split>/`.
