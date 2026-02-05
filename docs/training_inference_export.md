# Training, inference, and export

This note provides a minimal, end-to-end path for training, inference, and exporting predictions.

## Training (RT-DETR pose scaffold)

1) Install dependencies (CPU PyTorch for local dev):
- python3 -m pip install -r requirements-test.txt

2) Fetch the sample dataset (coco128):
- bash tools/fetch_coco128.sh

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

Plot loss curve (requires matplotlib):
- python3 tools/plot_metrics.py --jsonl reports/train_metrics.jsonl --out reports/train_loss.png

## Inference (adapter run)

Use the adapter tools to run inference and produce predictions JSON.

- python3 tools/export_predictions.py --adapter rtdetr_pose --config rtdetr_pose/configs/base.json --checkpoint /path/to.ckpt --max-images 50 --wrap --output reports/predictions.json

Optional TTA:
- python3 tools/export_predictions.py --adapter rtdetr_pose --tta --tta-seed 0 --tta-flip-prob 0.5 --wrap --output reports/predictions_tta.json

Note: TTA here is a lightweight **prediction-space transform** (a post-transform on the exported bboxes). It does not rerun the model on augmented inputs.

Optional TTT (test-time training, pre-prediction):
- Tent (entropy minimization):
	- python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-method tent --ttt-steps 5 --ttt-lr 1e-4 --wrap --output reports/predictions_ttt_tent.json
- MIM (masked image modeling):
	- python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-method mim --ttt-steps 5 --ttt-mask-prob 0.6 --ttt-patch-size 16 --wrap --output reports/predictions_ttt_mim.json

Notes:
- TTT requires an adapter that supports `get_model()` + `build_loader()` and requires torch.
- TTT updates model parameters in-memory before calling `adapter.predict(records)`.

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
