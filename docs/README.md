# YOLOZU docs

Choose one of these 4 entry points.
This page is intentionally an index-only gateway.
Keep deep procedural detail in the chapter docs/manual to avoid duplication.

## 0) Copy-paste smoke check (offline, repo checkout)

- Run: `bash scripts/smoke.sh`
- Bundled assets: `data/smoke`
- Report output: `reports/smoke_coco_eval_dry_run.json`

## 1) Evaluate from precomputed predictions (no inference deps)

- [External inference backends](external_inference.md)
- [Predictions schema](predictions_schema.md)

## 2) Train → Export → Eval (RT-DETR scaffold)

- [Training / inference / export](training_inference_export.md)
- [Run contract](run_contract.md)

## 3) Contracts (predictions schema / adapter contract / ttt protocol)

- [Predictions schema](predictions_schema.md)
- [Adapter contract](adapter_contract.md)
- [TTT protocol](ttt_protocol.md)

## 4) Bench/Parity (tensorrt_pipeline / benchmark_latency)

- [TensorRT pipeline](tensorrt_pipeline.md)
- [Benchmark latency](benchmark_latency.md)
