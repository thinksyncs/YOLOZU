# Training, inference, and export

This note provides a minimal, end-to-end path for training, inference, and exporting predictions.

## Training (RT-DETR pose scaffold)

1) Install dependencies (CPU PyTorch for local dev):
- python3 -m pip install -r requirements-test.txt

2) Fetch the sample dataset (coco128):
- bash tools/fetch_coco128.sh

Optional: prepare full COCO (val2017/train2017) in YOLO-format (recommended path under `data/`, which is gitignored):

```bash
# Example COCO layout:
#   /data/coco/
#     annotations/instances_val2017.json
#     val2017/*.jpg
python3 tools/prepare_coco_yolo.py \
  --coco-root /data/coco \
  --split val2017 \
  --out data/coco-yolo
```

Notes:
- `tools/prepare_coco_yolo.py` writes `data/coco-yolo/dataset.json` with absolute paths so evaluators can find images without copying.
- To copy images into `data/coco-yolo/images/<split>/`, add `--copy-images` (slower, larger).

3) Run the minimal trainer:
- python3 rtdetr_pose/tools/train_minimal.py --dataset-root data/coco128 --config rtdetr_pose/configs/base.json --max-steps 50 --use-matcher

Common options:
- --device cuda:0
- --batch-size 4
- --num-queries 10
- --stage-off-steps 1000 --stage-k-steps 1000
- --cost-z 1.0 --cost-rot 1.0 --cost-t 1.0
- --cost-z-start-step 500 --cost-rot-start-step 1000 --cost-t-start-step 1500
- --checkpoint-out reports/rtdetr_pose_ckpt.pt
- --metrics-jsonl reports/train_metrics.jsonl
- --metrics-csv reports/train_metrics.csv
- --tensorboard-logdir reports/tb

### Optimizer options

Choose between SGD and AdamW optimizers with configurable learning rates and weight decay:

```bash
# AdamW (default)
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --optimizer adamw \
  --lr 1e-4 \
  --weight-decay 0.01

# SGD with momentum and Nesterov
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --optimizer sgd \
  --lr 0.1 \
  --momentum 0.9 \
  --nesterov \
  --weight-decay 1e-4

# Use parameter groups with different lr/wd for backbone vs head
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --optimizer adamw \
  --lr 1e-4 \
  --use-param-groups \
  --backbone-lr-mult 0.1 \
  --head-lr-mult 1.0 \
  --backbone-wd-mult 0.5 \
  --head-wd-mult 1.0
```

### Learning rate scheduler options

Multiple scheduler types are supported with optional warmup:

```bash
# Cosine annealing with warmup
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --lr-warmup-steps 500 \
  --lr-warmup-init 1e-6

# OneCycleLR for super-convergence
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --scheduler onecycle

# MultiStepLR with decay at specific steps
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --scheduler multistep \
  --scheduler-milestones 1000,2000,3000 \
  --scheduler-gamma 0.1

# Linear schedule (legacy)
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --scheduler linear \
  --min-lr 1e-6
```

### Advanced training options

```bash
# Gradient clipping
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --clip-grad-norm 1.0

# Gradient accumulation (effective batch size = batch_size * gradient_accumulation_steps)
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --batch-size 2 \
  --gradient-accumulation-steps 4

# Automatic Mixed Precision (requires CUDA)
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --device cuda:0 \
  --use-amp

# Exponential Moving Average (EMA) of model weights
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --use-ema \
  --ema-decay 0.999 \
  --ema-eval  # Use EMA weights for evaluation/export

# Combined example: SGD + cosine scheduler + param groups + EMA
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --optimizer sgd \
  --momentum 0.9 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --lr-warmup-steps 500 \
  --use-param-groups \
  --backbone-lr-mult 0.1 \
  --use-ema \
  --ema-decay 0.999 \
  --clip-grad-norm 1.0
```

Plot loss curve (requires matplotlib):
- python3 tools/plot_metrics.py --jsonl reports/train_metrics.jsonl --out reports/train_loss.png

## Inference (adapter run)

Use the adapter tools to run inference and produce predictions JSON.

- python3 tools/export_predictions.py --adapter rtdetr_pose --config rtdetr_pose/configs/base.json --checkpoint /path/to.ckpt --max-images 50 --wrap --output reports/predictions.json

Optional TTA:
- python3 tools/export_predictions.py --adapter rtdetr_pose --tta --tta-seed 0 --tta-flip-prob 0.5 --wrap --output reports/predictions_tta.json

Note: TTA here is a lightweight **prediction-space transform** (a post-transform on the exported bboxes). It does not rerun the model on augmented inputs.

Optional TTT (test-time training, pre-prediction):
- Tent (recommended safe preset + guard rails):
	- python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-preset safe --ttt-reset sample --wrap --output reports/predictions_ttt_safe.json
- MIM (recommended safe preset + guard rails):
	- python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-preset mim_safe --ttt-reset sample --wrap --output reports/predictions_ttt_mim_safe.json

Notes:
- TTT requires an adapter that supports `get_model()` + `build_loader()` and requires torch.
- TTT updates model parameters in-memory before calling `adapter.predict(records)`.
- Recommended comparison protocol and more examples: `docs/ttt_protocol.md`.

## Export predictions for evaluation

If you run inference externally (PyTorch/TensorRT/ONNX), export to the YOLOZU predictions schema.
Then validate and evaluate in this repo.

- python3 tools/validate_predictions.py reports/predictions.json
- python3 tools/eval_coco.py --dataset data/coco128 --predictions reports/predictions.json --bbox-format cxcywh_norm --max-images 50

## Scenario suite (local evaluation)

- python3 tools/run_scenarios.py --adapter precomputed --predictions reports/predictions.json --max-images 50

## Notes
- When using GPU, install CUDA-enabled PyTorch and use --device cuda:0.
- Keep the predictions schema consistent with the adapter output: image path + detections list.

## YOLO26n smoke (RT-DETR scaffold)

This repo includes a tiny “it runs end-to-end” smoke command that:
- fetches `data/coco128` if missing
- runs a few steps of `rtdetr_pose/tools/train_minimal.py`
- exports `model.onnx`
- exports wrapped predictions JSON
- runs `tools/eval_suite.py --dry-run` to validate the full I/O chain

```bash
python3 tools/run_yolo26n_smoke_rtdetr_pose.py
```
