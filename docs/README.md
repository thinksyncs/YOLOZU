# YOLOZU docs

YOLOZU is an Apache-2.0-only, **contract-first evaluation harness**.
This `docs/` folder is organized by **what you want to do**.

Start with a concrete artifact:
- [Proof (one page): shortest path + report shape](proof_onepager.md)

## Choose an entry point

### 1) Evaluate-only (BYO inference)

You run inference anywhere. YOLOZU validates and scores a stable `predictions.json`.

Shortest path (pip):

```bash
yolozu validate predictions --predictions predictions.json --strict
yolozu eval-coco --dataset /path/to/coco --split val2017 --predictions predictions.json --output reports/coco_eval.json
```

Read next:
- [External inference backends](external_inference.md)
- [Predictions schema (stable contract)](predictions_schema.md)
- [Adapter strategy (priorities + workflow)](adapter_strategy.md)

### 2) Train/Export (RT-DETR pose)

Train with reproducible artifacts, export ONNX, and (optionally) build TensorRT with parity checks.

Shortest command (repo smoke):

```bash
python3 tools/run_yolo26n_smoke_rtdetr_pose.py
```

Read next:
- [Training / inference / export](training_inference_export.md)
- [Run contract (training artifacts)](run_contract.md)
- [TensorRT pipeline](tensorrt_pipeline.md)

### 3) Online adaptation (TTA/TTT, Safe TTT)

For online adaptation, YOLOZU focuses on **safe-by-default** update scopes + reset policies.

Read next:
- [TTT protocol (guard rails / reset)](ttt_protocol.md)
- [TTT/TTA support matrix](tta_support_matrix.md)
- [CoTTA phase-1 design spec](cotta_design_spec.md)

## Sharp edges (1-line)

- **Safe TTT**: guard rails + reset policies for Tent/MIM/CoTTA/EATA/SAR → [ttt_protocol.md](ttt_protocol.md)
- **Hessian refine**: engine-external post-processing over `predictions.json` → [hessian_solver.md](hessian_solver.md)
- **Depth mode**: `--depth-mode none|sidecar|fuse_mid` in RT-DETR pose scaffold → [training_inference_export.md](training_inference_export.md)

Other index:
- [Tools index](tools_index.md)
