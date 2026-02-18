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

## Full config schema (code-accurate)

Source of truth: `rtdetr_pose/tools/train_continual.py`.

Validation rules enforced by code:
- `schema_version` defaults to `1`; values other than `1` are rejected.
- `model_config` is required.
- `tasks` must be a non-empty list.
- each task must define `dataset_root`.
- `replay_fraction >= 0`, `replay_per_task_cap >= 0`.

Top-level keys:

| Key | Type | Default | Notes |
|---|---|---|---|
| `schema_version` | int | `1` | only `1` supported |
| `model_config` | str | required | RTDETR pose model JSON path |
| `train` | object | `{}` | forwarded to `train_minimal.py` (snake_case keys) |
| `continual` | object | `{}` | CL-specific options |
| `tasks` | list | required | sequential tasks |

`train` block:
- Keys are forwarded as `--<key-with-dashes>` to `train_minimal.py`.
- `seed`, `dataset_root`, `split` are managed by the runner and removed from forwarded keys.
- Recommended reference for accepted keys/defaults: `docs/training_inference_export.md`.

`continual` block:

| Key | Type | Default | Notes |
|---|---|---|---|
| `seed` | int | `train.seed` or `0` | runner/global seed |
| `replay_size` | int | `50` | `0` disables replay |
| `replay_strategy` | str | `reservoir` | reported in metadata |
| `replay_fraction` | float/null | `null` | replay_k = `round(fraction * train_records)` |
| `replay_per_task_cap` | int/null | `null` | cap replay samples per past task |

`continual.distill` (memoryless baseline):

| Key | Type | Default |
|---|---|---|
| `enabled` | bool | `true` |
| `keys` | str | `logits,bbox` |
| `weight` | float | `1.0` |
| `temperature` | float | `1.0` |
| `kl` | str | `reverse` |

`continual.lora` (optional):

| Key | Type | Default |
|---|---|---|
| `enabled` | bool | `false` |
| `r` | int | `0` (effective when enabled) |
| `alpha` | float/null | `null` |
| `dropout` | float | `0.0` |
| `target` | str | `head` |
| `freeze_base` | bool | `true` |
| `train_bias` | str | `none` |

`continual.derpp` (optional):

| Key | Type | Default |
|---|---|---|
| `enabled` | bool | `false` |
| `teacher_key` | str | `derpp_teacher_npz` |
| `keys` | str | `logits,bbox` |
| `weight` | float | `1.0` |
| `temperature` | float | `1.0` |
| `kl` | str | `reverse` |
| `logits_weight` | float | `1.0` |
| `bbox_weight` | float | `1.0` |
| `other_l1_weight` | float | `1.0` |

`continual.ewc` / `continual.si` (optional):

| Key | Type | Default | Notes |
|---|---|---|---|
| `ewc.enabled` | bool | `false` | enables `--ewc` |
| `ewc.lambda` | float | unchanged trainer default (`1.0`) if omitted | passed as `--ewc-lambda` only when present |
| `si.enabled` | bool | `false` | enables `--si` |
| `si.c` | float | unchanged trainer default (`1.0`) if omitted | passed as `--si-c` only when present |
| `si.epsilon` | float | unchanged trainer default (`1e-3`) if omitted | passed as `--si-epsilon` only when present |

`tasks[]` items:

| Key | Type | Default | Notes |
|---|---|---|---|
| `name` | str | `taskXX` | used in output folder naming |
| `dataset_root` | str | required | YOLO-format root |
| `train_split` | str | `train2017` (`split` fallback) | training split |
| `val_split` | str | `val2017` | metadata/eval split tag |

### CLI overrides for the runner

`train_continual.py` runner-only flags:

| Flag | Default | Effect |
|---|---|---|
| `--config` | required | continual YAML/JSON |
| `--run-dir` | auto timestamp dir | output base dir |
| `--replay-size` | `None` | overrides `continual.replay_size` |
| `--replay-fraction` | `None` | overrides `continual.replay_fraction` |
| `--replay-per-task-cap` | `None` | overrides `continual.replay_per_task_cap` |

## Notes / caveats

- The current continual evaluation uses `yolozu.simple_map` (CPU-friendly proxy). For full COCO mAP you can switch your workflow to `tools/eval_coco.py` with `pycocotools` installed.
- `rtdetr_pose` dataset loading scans `*.jpg`, `*.jpeg`, and `*.png` under `images/<split>/`.
