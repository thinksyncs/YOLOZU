# YOLOZU docs

Choose one of these 4 entry points.
Use `data/smoke` as the default copy-paste dataset in this docs index
to avoid path mistakes.

## A) Evaluate from precomputed predictions (no inference deps)

Shortest 3 commands:

```bash
python3 -m yolozu.cli validate dataset data/smoke --strict
python3 -m yolozu.cli validate predictions data/smoke/predictions/predictions_dummy.json --strict
python3 -m yolozu.cli eval-coco \
	--dataset data/smoke \
	--split val \
	--predictions data/smoke/predictions/predictions_dummy.json \
	--dry-run \
	--output reports/smoke_coco_eval_dry_run.json
```

Reference docs:
- [External inference backends](external_inference.md)
- [Predictions schema](predictions_schema.md)

## B) Train → Export → Eval (RT-DETR scaffold)

Shortest 3 commands (smoke-safe export/eval path):

```bash
python3 -m yolozu.cli validate dataset data/smoke --strict
python3 -m yolozu.cli export --backend labels --dataset data/smoke --output runs/smoke/predictions_labels.json --force
python3 -m yolozu.cli eval-coco \
	--dataset data/smoke \
	--split val \
	--predictions runs/smoke/predictions_labels.json \
	--dry-run \
	--output runs/smoke/coco_eval_dry_run.json
```

Reference docs:
- [Training / inference / export](training_inference_export.md)
- [Run contract](run_contract.md)

## C) Contracts (predictions / adapter / TTT protocol)

Shortest 3 commands:

```bash
python3 -m yolozu.cli validate predictions data/smoke/predictions/predictions_dummy.json --strict
python3 -m yolozu.cli validate dataset data/smoke --strict
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
```

Reference docs:
- [Predictions schema](predictions_schema.md)
- [Adapter contract](adapter_contract.md)
- [TTT protocol](ttt_protocol.md)

## D) Bench/Parity (parity check + latency benchmark docs)

Shortest 3 commands:

```bash
python3 -m yolozu.cli parity \
	--reference data/smoke/predictions/predictions_dummy.json \
	--candidate data/smoke/predictions/predictions_dummy.json
python3 -m yolozu.cli eval-coco \
	--dataset data/smoke \
	--split val \
	--predictions data/smoke/predictions/predictions_dummy.json \
	--dry-run \
	--output reports/smoke_parity_eval_dry_run.json
python3 tools/benchmark_latency.py --help
```

Reference docs:
- [TensorRT pipeline](tensorrt_pipeline.md)
- [Benchmark latency](benchmark_latency.md)

## 0) Copy-paste smoke check (offline, repo checkout)

- Run: `bash scripts/smoke.sh`
- Bundled assets: `data/smoke`
- Report output: `reports/smoke_coco_eval_dry_run.json`
